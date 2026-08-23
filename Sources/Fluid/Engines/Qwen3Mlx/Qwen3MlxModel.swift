import Foundation
import MLX
import MLXFast
import MLXNN

// Ported from https://github.com/gabrimatic/qwen3-asr-mlx (MIT) - MLX Qwen3-ASR
// inference for Apple Silicon, adapted to mlx-swift with dynamic spec support.
// Weights are injected at construction time (no update(parameters:) machinery):
// every module reads its weights from the flat snake_case table directly.

// MARK: - Helpers

func q3AskQuantized(_ key: String, w: [String: MLXArray], bias: Bool, quant: Qwen3MlxQuantInfo) -> Bool {
    w["\(key).scales"] != nil
}

func q3AsMakeLinear(_ key: String, w: [String: MLXArray], inDim: Int, outDim: Int, bias: Bool, quant: Qwen3MlxQuantInfo?) -> Module {
    if let quant, w["\(key).scales"] != nil, let wt = w["\(key).weight"], let sc = w["\(key).scales"] {
        return QuantizedLinear(
            weight: wt, bias: (bias ? w["\(key).bias"] : nil), scales: sc,
            biases: w["\(key).biases"], groupSize: quant.groupSize, bits: quant.bits)
    }
    guard let wt = w["\(key).weight"] else {
        return Linear(inDim, outDim, bias: bias)
    }
    return Linear(weight: wt, bias: (bias ? w["\(key).bias"] : nil))
}

// MARK: - KV Cache

final class Qwen3MlxKVCache {
    var keys: [MLXArray] = []
    var values: [MLXArray] = []
    var offset: Int = 0

    func update(layerIdx: Int, key: MLXArray, value: MLXArray) -> (MLXArray, MLXArray) {
        guard layerIdx < keys.count else {
            keys.append(key)
            values.append(value)
            return (key, value)
        }
        keys[layerIdx] = concatenated([keys[layerIdx], key], axis: 2)
        values[layerIdx] = concatenated([values[layerIdx], value], axis: 2)
        return (keys[layerIdx], values[layerIdx])
    }
}

// MARK: - Injectable base layers (MLXNN let-parameter layers replaced)

nonisolated final class Qwen3MlxLayerNorm: Module {
    var weight: MLXArray
    var bias: MLXArray?
    let eps: Float

    init(dimensions: Int, eps: Float = 1e-5) {
        self.weight = MLXArray.ones([dimensions])
        self.bias = MLXArray.zeros([dimensions])
        self.eps = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.layerNorm(x, weight: weight, bias: bias, eps: eps)
    }
}

nonisolated final class Qwen3MlxRMSNorm: Module {
    var weight: MLXArray
    let eps: Float

    init(dimensions: Int, eps: Float = 1e-6) {
        self.weight = MLXArray.ones([dimensions])
        self.eps = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Compute in float32: activations can exceed the fp16 square range
        // (e.g. Fun-ASR audio embeddings ~1e3+) and overflow to inf.
        let xf = x.asType(.float32)
        let mean = mean(xf * xf, axis: -1, keepDims: true)
        let rstd = rsqrt(mean + eps)
        return (weight.asType(.float32) * xf * rstd).asType(x.dtype)
    }
}

nonisolated final class Qwen3MlxEmbedding: Module {
    var weight: MLXArray
    var quantized = false
    var scales: MLXArray?
    var biases: MLXArray?
    var groupSize = 64
    var bits = 4

    init(embeddingCount: Int, dimensions: Int) {
        self.weight = MLXRandom.normal([embeddingCount, dimensions])
        super.init()
    }

    func effectiveWeight() -> MLXArray {
        guard quantized, let scales else { return weight }
        return dequantized(weight, scales: scales, biases: biases, groupSize: groupSize, bits: bits)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let w = effectiveWeight()
        let flat = x.flattened()
        let indices = flat.asType(.int32)
        let out = take(w, indices, axis: 0)
        if x.ndim > 1 {
            return out.reshaped(x.shape + [w.shape[1]])
        }
        return out
    }
}

nonisolated final class Qwen3MlxConvStem: Module {
    var weights: [MLXArray] = []
    var biases: [MLXArray?] = []

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = x
        for i in 0..<weights.count {
            let z = conv2d(y, weights[i], stride: 2, padding: 1)
            y = gelu(biases[i] != nil ? z + biases[i]! : z)
        }
        return y
    }
}

// MARK: - Encoder (Whisper-style)

nonisolated final class Qwen3MlxEncoderLayer: Module {
    let selfAttnNorm: Qwen3MlxLayerNorm
    let selfAttn: Qwen3MlxEncoderAttention
    let finalNorm: Qwen3MlxLayerNorm
    let fc1: Module
    let fc2: Module

    init(spec: Qwen3MlxSpec, prefix: String, w: [String: MLXArray], quant: Qwen3MlxQuantInfo?) {
        self.selfAttnNorm = Qwen3MlxLayerNorm(dimensions: spec.encDModel)
        self.finalNorm = Qwen3MlxLayerNorm(dimensions: spec.encDModel)
        if let nw = w["\(prefix)self_attn_layer_norm.weight"] {
            self.selfAttnNorm.weight = nw
            if let nb = w["\(prefix)self_attn_layer_norm.bias"] { self.selfAttnNorm.bias = nb }
        }
        if let nw = w["\(prefix)final_layer_norm.weight"] {
            self.finalNorm.weight = nw
            if let nb = w["\(prefix)final_layer_norm.bias"] { self.finalNorm.bias = nb }
        }
        self.selfAttn = Qwen3MlxEncoderAttention(spec: spec, prefix: prefix, w: w, quant: quant)
        self.fc1 = q3AsMakeLinear("\(prefix)fc1", w: w, inDim: spec.encDModel, outDim: spec.encFFNDim, bias: true, quant: quant)
        self.fc2 = q3AsMakeLinear("\(prefix)fc2", w: w, inDim: spec.encFFNDim, outDim: spec.encDModel, bias: true, quant: quant)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        let residual = x
        var h = selfAttnNorm(x)
        h = selfAttn(h, mask: mask)
        h = residual + h
        let residual2 = h
        h = finalNorm(h)
        h = gelu((fc1 as! UnaryLayer)(h))
        h = (fc2 as! UnaryLayer)(h)
        return residual2 + h
    }
}

nonisolated final class Qwen3MlxEncoderAttention: Module {
    let heads: Int
    let headDim: Int
    let scale: Float
    let qProj: Module
    let kProj: Module
    let vProj: Module
    let outProj: Module

    init(spec: Qwen3MlxSpec, prefix: String, w: [String: MLXArray], quant: Qwen3MlxQuantInfo?) {
        self.heads = spec.encHeads
        self.headDim = spec.encDModel / spec.encHeads
        self.scale = powf(Float(self.headDim), -0.5)
        let d = spec.encDModel
        self.qProj = q3AsMakeLinear("\(prefix)self_attn.q_proj", w: w, inDim: d, outDim: d, bias: true, quant: quant)
        self.kProj = q3AsMakeLinear("\(prefix)self_attn.k_proj", w: w, inDim: d, outDim: d, bias: true, quant: quant)
        self.vProj = q3AsMakeLinear("\(prefix)self_attn.v_proj", w: w, inDim: d, outDim: d, bias: true, quant: quant)
        self.outProj = q3AsMakeLinear("\(prefix)self_attn.out_proj", w: w, inDim: d, outDim: d, bias: true, quant: quant)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        let shape = x.shape
        let b = shape[0]
        let t = shape[1]
        let h = heads
        let d = headDim

        let q = (qProj as! UnaryLayer)(x).reshaped([b, t, h, d]).transposed(0, 2, 1, 3)
        let k = (kProj as! UnaryLayer)(x).reshaped([b, t, h, d]).transposed(0, 2, 1, 3)
        let v = (vProj as! UnaryLayer)(x).reshaped([b, t, h, d]).transposed(0, 2, 1, 3)

        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: mask
        )
        let merged = out.transposed(0, 2, 1, 3).reshaped([b, t, h * d])
        return (outProj as! UnaryLayer)(merged)
    }
}

nonisolated final class Qwen3MlxAudioEncoder: Module {
    let spec: Qwen3MlxSpec
    let convStem: Qwen3MlxConvStem
    let convOut: Module
    let positionalEmbedding: MLXArray
    let layers: [Qwen3MlxEncoderLayer]
    let lnPost: Qwen3MlxLayerNorm
    let proj1: Module
    let proj2: Module

    init(spec: Qwen3MlxSpec, w: [String: MLXArray], quant: Qwen3MlxQuantInfo?) {
        self.spec = spec
        let stem = Qwen3MlxConvStem()
        for i in 1...3 {
            if let cw = w["conv2d\(i).weight"] {
                stem.weights.append(cw)
                stem.biases.append(w["conv2d\(i).bias"])
            }
        }
        self.convStem = stem
        let freqAfterConv = ((((spec.numMelBins + 1) / 2 + 1) / 2 + 1) / 2)
        self.convOut = q3AsMakeLinear("conv_out", w: w, inDim: spec.downsampleHidden * freqAfterConv, outDim: spec.encDModel, bias: false, quant: quant)
        self.positionalEmbedding = Self.sinusoidalPositions(
            maxPositions: spec.maxSourcePositions, dModel: spec.encDModel)
        self.layers = (0..<spec.encLayers).map { i in
            Qwen3MlxEncoderLayer(spec: spec, prefix: "layers.\(i).", w: w, quant: quant)
        }
        let lp = Qwen3MlxLayerNorm(dimensions: spec.encDModel)
        if let nw = w["ln_post.weight"] {
            lp.weight = nw
            if let nb = w["ln_post.bias"] { lp.bias = nb }
        }
        self.lnPost = lp
        self.proj1 = q3AsMakeLinear("proj1", w: w, inDim: spec.encDModel, outDim: spec.encDModel, bias: true, quant: quant)
        self.proj2 = q3AsMakeLinear("proj2", w: w, inDim: spec.encDModel, outDim: spec.encOutputDim, bias: true, quant: quant)
        super.init()
    }

    static func sinusoidalPositions(maxPositions: Int, dModel: Int) -> MLXArray {
        let half = dModel / 2
        let logTimescale = log(10_000.0) / Double(half - 1)
        let invTimescales = exp(-MLXArray(0..<half).asType(.float32) * Float(logTimescale))
        let positions = MLXArray(0..<maxPositions).asType(.float32)
        let scaled = positions[0..., .newAxis] * invTimescales[.newAxis, 0...]
        return concatenated([sin(scaled), cos(scaled)], axis: 1)
    }

    static func convOutputLength(_ inputLength: Int) -> Int {
        var l = inputLength
        for _ in 0..<3 { l = (l - 1) / 2 + 1 }
        return l
    }

    static func blockAttentionMask(seqLen: Int, cuSeqLens: [Int]) -> MLXArray? {
        if cuSeqLens.count <= 2 { return nil }
        var data = [Float](repeating: 0, count: seqLen * seqLen)
        for i in 0..<(cuSeqLens.count - 1) {
            let start = cuSeqLens[i]
            let end = cuSeqLens[i + 1]
            if end > start {
                var r = start
                while r < end {
                    var c = start
                    while c < end {
                        data[r * seqLen + c] = 1
                        c += 1
                    }
                    r += 1
                }
            }
        }
        let mask = which(MLXArray(data).reshaped([seqLen, seqLen]), 0.0, -1e9)
        return mask.reshaped([1, 1, seqLen, seqLen])
    }

    func callAsFunction(_ mel: MLXArray) -> MLXArray {
        var m = mel
        if m.ndim == 3 { m = m[0] }
        let nMel = m.shape[0]
        let tLen = m.shape[1]
        let chunkSize = spec.chunkSize

        var chunkMels: [MLXArray] = []
        var chunkRealLengths: [Int] = []
        var offset = 0
        while offset < tLen {
            let end = min(offset + chunkSize, tLen)
            let segment = m[0..., offset..<end]
            chunkRealLengths.append(end - offset)
            if end - offset < chunkSize {
                let pad = zeros([nMel, chunkSize - (end - offset)])
                chunkMels.append(concatenated([segment, pad], axis: 1))
            } else {
                chunkMels.append(segment)
            }
            offset += chunkSize
        }

        // Conv input layout is channels-last: [numChunks, nMel, chunk, 1].
        let batched = stacked(chunkMels, axis: 0).expandedDimensions(axis: -1)
        let xConv = convStem(batched)  // [numChunks, freq, time, channels]

        let convShape = xConv.shape  // [numChunks, melFreq, time, channels]
        let b = convShape[0]
        let freq = convShape[1]
        let time = convShape[2]
        let ch = convShape[3]
        let x = (convOut as! UnaryLayer)(
            xConv.transposed(0, 2, 3, 1).reshaped([b, time, ch * freq]))

        let tokensPerChunk = x.shape[1]
        let pe = positionalEmbedding[0..<tokensPerChunk, 0...]
        let xPos = x + pe

        let validLengths = chunkRealLengths.map { Self.convOutputLength($0) }
        var hiddenList: [MLXArray] = []
        for i in 0..<chunkMels.count {
            hiddenList.append(xPos[i, 0..<validLengths[i], 0...])
        }
        let hidden = concatenated(hiddenList, axis: 0)

        let totalTokens = hidden.shape[0]
        let windowTokens = tokensPerChunk * (spec.nWindowInfer / chunkSize)
        var cuSeqLens = [0]
        let numFullWindows = totalTokens / windowTokens
        for _ in 0..<numFullWindows {
            cuSeqLens.append(cuSeqLens[cuSeqLens.count - 1] + windowTokens)
        }
        let remainder = totalTokens % windowTokens
        if remainder > 0 {
            cuSeqLens.append(cuSeqLens[cuSeqLens.count - 1] + remainder)
        }
        let mask = Self.blockAttentionMask(seqLen: totalTokens, cuSeqLens: cuSeqLens)

        var h = hidden[.newAxis, 0..., 0...]
        for layer in layers {
            h = layer(h, mask: mask)
        }

        h = lnPost(h)
        h = gelu((proj1 as! UnaryLayer)(h))
        h = (proj2 as! UnaryLayer)(h)
        return h
    }
}

// MARK: - Decoder (Qwen3 LLM)

nonisolated final class Qwen3MlxMLP: Module {
    let gateProj: Module
    let upProj: Module
    let downProj: Module

    init(spec: Qwen3MlxSpec, prefix: String, w: [String: MLXArray], quant: Qwen3MlxQuantInfo?) {
        let h = spec.hiddenSize
        let inter = spec.intermediateSize
        self.gateProj = q3AsMakeLinear("\(prefix)gate_proj", w: w, inDim: h, outDim: inter, bias: false, quant: quant)
        self.upProj = q3AsMakeLinear("\(prefix)up_proj", w: w, inDim: h, outDim: inter, bias: false, quant: quant)
        self.downProj = q3AsMakeLinear("\(prefix)down_proj", w: w, inDim: inter, outDim: h, bias: false, quant: quant)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let g = silu((gateProj as! UnaryLayer)(x))
        let u = (upProj as! UnaryLayer)(x)
        return (downProj as! UnaryLayer)(g * u)
    }
}

nonisolated final class Qwen3MlxAttention: Module {
    let nHeads: Int
    let nKVHeads: Int
    let headDim: Int
    let scale: Float
    let qProj: Module
    let kProj: Module
    let vProj: Module
    let oProj: Module
    let qNorm: Qwen3MlxRMSNorm
    let kNorm: Qwen3MlxRMSNorm
    let rope: RoPE

    init(spec: Qwen3MlxSpec, prefix: String, w: [String: MLXArray], quant: Qwen3MlxQuantInfo?) {
        nHeads = spec.numHeads
        nKVHeads = spec.numKVHeads
        headDim = spec.headDim
        scale = powf(Float(spec.headDim), -0.5)
        let h = spec.hiddenSize
        qProj = q3AsMakeLinear("\(prefix)q_proj", w: w, inDim: h, outDim: nHeads * headDim, bias: false, quant: quant)
        kProj = q3AsMakeLinear("\(prefix)k_proj", w: w, inDim: h, outDim: nKVHeads * headDim, bias: false, quant: quant)
        vProj = q3AsMakeLinear("\(prefix)v_proj", w: w, inDim: h, outDim: nKVHeads * headDim, bias: false, quant: quant)
        oProj = q3AsMakeLinear("\(prefix)o_proj", w: w, inDim: nHeads * headDim, outDim: h, bias: false, quant: quant)
        qNorm = Qwen3MlxRMSNorm(dimensions: headDim, eps: spec.rmsEps)
        kNorm = Qwen3MlxRMSNorm(dimensions: headDim, eps: spec.rmsEps)
        if let qw = w["\(prefix)q_norm.weight"] { qNorm.weight = qw }
        if let kw = w["\(prefix)k_norm.weight"] { kNorm.weight = kw }
        rope = RoPE(dimensions: headDim, traditional: false, base: spec.ropeTheta)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cache: Qwen3MlxKVCache?, layerIdx: Int) -> MLXArray {
        let shape = x.shape
        let b = shape[0]
        let t = shape[1]

        var q = (qProj as! UnaryLayer)(x).reshaped([b, t, nHeads, headDim])
        var k = (kProj as! UnaryLayer)(x).reshaped([b, t, nKVHeads, headDim])
        let v = (vProj as! UnaryLayer)(x).reshaped([b, t, nKVHeads, headDim])

        q = qNorm(q)
        k = kNorm(k)

        q = q.transposed(0, 2, 1, 3)
        k = k.transposed(0, 2, 1, 3)
        let vT = v.transposed(0, 2, 1, 3)

        let offset = cache?.offset ?? 0
        q = rope(q, offset: offset)
        k = rope(k, offset: offset)

        let kFull: MLXArray
        let vFull: MLXArray
        if let cache {
            let updated = cache.update(layerIdx: layerIdx, key: k, value: vT)
            kFull = updated.0
            vFull = updated.1
        } else {
            kFull = k
            vFull = vT
        }

        let mask = createAdditiveCausalMask(n: t, offset: offset).asType(q.dtype)
        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: kFull, values: vFull, scale: scale, mask: mask
        )
        let merged = out.transposed(0, 2, 1, 3).reshaped([b, t, nHeads * headDim])
        let proj = (oProj as! UnaryLayer)(merged)
        return proj
    }
}

func createAdditiveCausalMask(n: Int, offset: Int = 0) -> MLXArray {
    let rInds = MLXArray(0..<(offset + n)).asType(.float32)
    let lInds = offset > 0 ? MLXArray(offset..<(offset + n)).asType(.float32) : rInds
    // out[i][j] = lInds[i] < rInds[j]  (future j > i masked)
    let maskBase = (lInds[0..., .newAxis] .< rInds[.newAxis, 0...]) * -1e9
    return maskBase.reshaped([1, 1, n, offset + n])
}

nonisolated final class Qwen3MlxDecoderLayer: Module {
    let inputLayerNorm: Qwen3MlxRMSNorm
    let selfAttn: Qwen3MlxAttention
    let postAttentionLayerNorm: Qwen3MlxRMSNorm
    let mlp: Qwen3MlxMLP

    init(spec: Qwen3MlxSpec, prefix: String, w: [String: MLXArray], quant: Qwen3MlxQuantInfo?) {
        inputLayerNorm = Qwen3MlxRMSNorm(dimensions: spec.hiddenSize, eps: spec.rmsEps)
        postAttentionLayerNorm = Qwen3MlxRMSNorm(dimensions: spec.hiddenSize, eps: spec.rmsEps)
        if let nw = w["\(prefix)input_layernorm.weight"] { inputLayerNorm.weight = nw }
        if let nw = w["\(prefix)post_attention_layernorm.weight"] { postAttentionLayerNorm.weight = nw }
        selfAttn = Qwen3MlxAttention(spec: spec, prefix: "\(prefix)self_attn.", w: w, quant: quant)
        mlp = Qwen3MlxMLP(spec: spec, prefix: "\(prefix)mlp.", w: w, quant: quant)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cache: Qwen3MlxKVCache?, layerIdx: Int) -> MLXArray {
        let residual = x
        var h = inputLayerNorm(x)
        h = selfAttn(h, cache: cache, layerIdx: layerIdx)
        h = residual + h
        let residual2 = h
        h = postAttentionLayerNorm(h)
        h = mlp(h)
        return residual2 + h
    }
}

nonisolated final class Qwen3MlxTextDecoder: Module {
    let spec: Qwen3MlxSpec
    let embedTokens: Qwen3MlxEmbedding
    let layers: [Qwen3MlxDecoderLayer]
    let norm: Qwen3MlxRMSNorm

    init(spec: Qwen3MlxSpec, w: [String: MLXArray], quant: Qwen3MlxQuantInfo?) {
        self.spec = spec
        let emb = Qwen3MlxEmbedding(
            embeddingCount: spec.vocabSize, dimensions: spec.hiddenSize)
        if let ew = w["embed_tokens.weight"] {
            emb.weight = ew
            if let es = w["embed_tokens.scales"] {
                emb.scales = es
                emb.biases = w["embed_tokens.biases"]
                emb.groupSize = quant?.groupSize ?? 64
                emb.bits = quant?.bits ?? 4
                emb.quantized = true
            }
        }
        self.embedTokens = emb
        self.layers = (0..<spec.numLayers).map { i in
            Qwen3MlxDecoderLayer(spec: spec, prefix: "layers.\(i).", w: w, quant: quant)
        }
        let n = Qwen3MlxRMSNorm(dimensions: spec.hiddenSize, eps: spec.rmsEps)
        if let nw = w["norm.weight"] { n.weight = nw }
        self.norm = n
        super.init()
    }

    /// Untied lm-head weight (nil = tied to embed tokens).
    var lmHeadWeight: MLXArray?
    var lmHeadScales: MLXArray?
    var lmHeadBiases: MLXArray?
    var lmHeadGroupSize = 64
    var lmHeadBits = 8

    func callAsFunction(_ inputs: MLXArray, cache: Qwen3MlxKVCache?, isEmbeds: Bool) -> MLXArray {
        var h = isEmbeds ? inputs : embedTokens(inputs)
        for (i, layer) in layers.enumerated() {
            h = layer(h, cache: cache, layerIdx: i)
        }
        h = norm(h)
        if let lmHeadWeight {
            if let lmHeadScales {
                let w = dequantized(
                    lmHeadWeight, scales: lmHeadScales, biases: lmHeadBiases,
                    groupSize: lmHeadGroupSize, bits: lmHeadBits)
                return matmul(h, w.T)
            }
            // Plain (unquantized) untied head.
            return matmul(h, lmHeadWeight.T)
        }
        return matmul(h, embedTokens.effectiveWeight().T)
    }
}
