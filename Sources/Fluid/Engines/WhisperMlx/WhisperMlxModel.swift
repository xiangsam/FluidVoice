import Foundation
import MLX
import MLXNN

// MARK: - Whisper MLX model (ported from mlx_audio.stt.models.whisper)
//
// Encoder: conv1d stem (stride 2) + sinusoidal PE + N pre-norm Transformer
// layers (full attention, GELU MLP) + LN.
// Decoder: token + positional embedding + N layers with causal self-attention,
// cross-attention over the encoder output and GELU MLP + LN, tied lm head.

// MARK: - Helpers

private func whisperLinear(
    _ key: String, w: [String: MLXArray], inDim: Int, outDim: Int,
    bias: Bool, bits: Int, groupSize: Int
) -> Module {
    if let wt = w["\(key).weight"], let sc = w["\(key).scales"] {
        return QuantizedLinear(
            weight: wt, bias: (bias ? w["\(key).bias"] : nil), scales: sc,
            biases: w["\(key).biases"], groupSize: groupSize, bits: bits)
    }
    guard let wt = w["\(key).weight"] else {
        return Linear(inDim, outDim, bias: bias)
    }
    return Linear(weight: wt, bias: (bias ? w["\(key).bias"] : nil))
}

private func whisperSinusoids(length: Int, channels: Int) -> MLXArray {
    assert(channels % 2 == 0)
    let half = channels / 2
    let logInc = log(10_000.0) / Double(half - 1)
    let inv = exp(-MLXArray((0..<half)).asType(.float32) * Float(logInc)) // [half]
    let positions = MLXArray(0..<length).asType(.float32) // [length]
    let scaled = positions[0..., .newAxis] * inv[.newAxis, 0...] // [length, half]
    return concatenated([sin(scaled), cos(scaled)], axis: 1).asType(.float32)
}

// MARK: - Attention

nonisolated final class WhisperMlxAttention: Module {
    let heads: Int
    let dHead: Int
    let scale: Float
    let query: Module
    let key: Module
    let value: Module
    let out: Module

    init(prefix: String, w: [String: MLXArray], state: Int, heads: Int, bits: Int, groupSize: Int) {
        self.heads = heads
        self.dHead = state / heads
        // Whisper scales BOTH query and key by d_head^-0.25.
        self.scale = powf(Float(self.dHead), -0.25)
        self.query = whisperLinear("\(prefix)query", w: w, inDim: state, outDim: state, bias: true, bits: bits, groupSize: groupSize)
        self.key = whisperLinear("\(prefix)key", w: w, inDim: state, outDim: state, bias: false, bits: bits, groupSize: groupSize)
        self.value = whisperLinear("\(prefix)value", w: w, inDim: state, outDim: state, bias: true, bits: bits, groupSize: groupSize)
        self.out = whisperLinear("\(prefix)out", w: w, inDim: state, outDim: state, bias: true, bits: bits, groupSize: groupSize)
        super.init()
    }

    /// x [1, T, D]. xa nil -> self-attention; else cross-attention.
    /// Returns (output, (k, v), qk).
    func callAsFunction(_ x: MLXArray, xa: MLXArray?, mask: MLXArray?, kv: (MLXArray, MLXArray)?)
        -> (MLXArray, (MLXArray, MLXArray), MLXArray?)
    {
        let b = x.shape[0]
        let t = x.shape[1]
        var q = (self.query as! UnaryLayer)(x)
        let kRaw: MLXArray
        let vRaw: MLXArray
        if xa == nil {
            let kNew = (self.key as! UnaryLayer)(x)
            let vNew = (self.value as! UnaryLayer)(x)
            if let existing = kv {
                kRaw = concatenated([existing.0, kNew], axis: 1)
                vRaw = concatenated([existing.1, vNew], axis: 1)
            } else {
                kRaw = kNew
                vRaw = vNew
            }
        } else if kv == nil {
            kRaw = (self.key as! UnaryLayer)(xa!)
            vRaw = (self.value as! UnaryLayer)(xa!)
        } else {
            kRaw = kv!.0
            vRaw = kv!.1
        }

        // Cache is kept in the un-transposed [b, t, d] layout so consecutive
        // steps concatenate along the sequence axis.
        let total = kRaw.shape[1]
        q = q.reshaped([b, t, heads, dHead]).transposed(0, 2, 1, 3) * scale
        let k = kRaw.reshaped([b, total, heads, dHead]).transposed(0, 2, 3, 1) * scale
        let v = vRaw.reshaped([b, total, heads, dHead]).transposed(0, 2, 1, 3)

        var qk = matmul(q, k) // [b, heads, t, total]
        if let mask {
            qk = qk + mask
        }
        let attn = softMax(qk, axis: -1)
        let outArr = matmul(attn, v)
        let merged = outArr.transposed(0, 2, 1, 3).reshaped([b, t, heads * dHead])
        return ((self.out as! UnaryLayer)(merged), (kRaw, vRaw), qk)
    }
}

// MARK: - Residual block

nonisolated final class WhisperMlxBlock: Module {
    let attn: WhisperMlxAttention
    let attnLN: Qwen3MlxLayerNorm
    let crossAttn: WhisperMlxAttention?
    let crossAttnLN: Qwen3MlxLayerNorm?
    let mlp1: Module
    let mlp2: Module
    let mlpLN: Qwen3MlxLayerNorm
    let state: Int
    let heads: Int

    init(prefix: String, w: [String: MLXArray], state: Int, heads: Int, cross: Bool, bits: Int, groupSize: Int) {
        self.state = state
        self.heads = heads
        self.attn = WhisperMlxAttention(prefix: "\(prefix)attn.", w: w, state: state, heads: heads, bits: bits, groupSize: groupSize)
        let a = Qwen3MlxLayerNorm(dimensions: state)
        if let nw = w["\(prefix)attn_ln.weight"] { a.weight = nw }
        if let nb = w["\(prefix)attn_ln.bias"] { a.bias = nb }
        self.attnLN = a
        if cross {
            self.crossAttn = WhisperMlxAttention(prefix: "\(prefix)cross_attn.", w: w, state: state, heads: heads, bits: bits, groupSize: groupSize)
            let c = Qwen3MlxLayerNorm(dimensions: state)
            if let nw = w["\(prefix)cross_attn_ln.weight"] { c.weight = nw }
            if let nb = w["\(prefix)cross_attn_ln.bias"] { c.bias = nb }
            self.crossAttnLN = c
        } else {
            self.crossAttn = nil
            self.crossAttnLN = nil
        }
        self.mlp1 = whisperLinear("\(prefix)mlp1", w: w, inDim: state, outDim: state * 4, bias: true, bits: bits, groupSize: groupSize)
        self.mlp2 = whisperLinear("\(prefix)mlp2", w: w, inDim: state * 4, outDim: state, bias: true, bits: bits, groupSize: groupSize)
        let m = Qwen3MlxLayerNorm(dimensions: state)
        if let nw = w["\(prefix)mlp_ln.weight"] { m.weight = nw }
        if let nb = w["\(prefix)mlp_ln.bias"] { m.bias = nb }
        self.mlpLN = m
        super.init()
    }

    /// Returns (output, (selfKV, crossKV)).
    func callAsFunction(
        _ x: MLXArray, xa: MLXArray?, mask: MLXArray?, kv: ((MLXArray, MLXArray)?, (MLXArray, MLXArray)?)?
    ) -> (MLXArray, ((MLXArray, MLXArray)?, (MLXArray, MLXArray)?)) {
        var h = x
        let (y, newKV, _) = self.attn(self.attnLN(h), xa: nil, mask: mask, kv: kv?.0)
        h = h + y
        var newCross: (MLXArray, MLXArray)? = kv?.1
        if let crossAttn, let crossAttnLN, let xa {
            let (cy, ckv, _) = crossAttn(crossAttnLN(h), xa: xa, mask: nil, kv: kv?.1)
            h = h + cy
            newCross = ckv
        }
        h = h + (self.mlp2 as! UnaryLayer)(gelu((self.mlp1 as! UnaryLayer)(self.mlpLN(h))))
        return (h, (newKV, newCross))
    }
}

// MARK: - Whisper model

nonisolated final class WhisperMlxModel: Module {
    let spec: WhisperMlxSpec
    let conv1Weight: MLXArray
    let conv1Bias: MLXArray?
    let conv2Weight: MLXArray
    let conv2Bias: MLXArray?
    let positionalEmbedding: MLXArray // encoder sinusoids [audioCtx, state]
    let encoderBlocks: [WhisperMlxBlock]
    let lnPost: Qwen3MlxLayerNorm

    let tokenEmbedding: Qwen3MlxEmbedding
    let decoderPE: MLXArray // [textCtx, state]
    private var cachedEmbedding: MLXArray?
    let decoderBlocks: [WhisperMlxBlock]
    let decoderLN: Qwen3MlxLayerNorm

    init(spec: WhisperMlxSpec, w: [String: MLXArray]) {
        self.spec = spec
        // Encoder conv1d weights are [O, K, C] in MLX format (HF repo transposed
        // at load time by remapWeights).
        self.conv1Weight = w["encoder.conv1.weight"] ?? .zeros([spec.audioState, 3, spec.nMels])
        self.conv1Bias = w["encoder.conv1.bias"]
        self.conv2Weight = w["encoder.conv2.weight"] ?? .zeros([spec.audioState, 3, spec.audioState])
        self.conv2Bias = w["encoder.conv2.bias"]
        self.positionalEmbedding = whisperSinusoids(length: spec.audioCtx, channels: spec.audioState)
        self.encoderBlocks = (0..<spec.audioLayers).map { i in
            WhisperMlxBlock(prefix: "encoder.blocks.\(i).", w: w, state: spec.audioState, heads: spec.audioHeads, cross: false, bits: spec.quantBits, groupSize: spec.quantGroupSize)
        }
        let lp = Qwen3MlxLayerNorm(dimensions: spec.audioState)
        if let nw = w["encoder.ln_post.weight"] { lp.weight = nw }
        if let nb = w["encoder.ln_post.bias"] { lp.bias = nb }
        self.lnPost = lp

        let emb = Qwen3MlxEmbedding(embeddingCount: spec.vocabSize, dimensions: spec.textState)
        if let ew = w["decoder.token_embedding.weight"] {
            emb.weight = ew
            if let es = w["decoder.token_embedding.scales"] {
                emb.scales = es
                emb.biases = w["decoder.token_embedding.biases"]
                emb.groupSize = spec.quantGroupSize
                emb.bits = spec.quantBits
                emb.quantized = true
            }
        }
        self.tokenEmbedding = emb
        self.decoderPE = (w["decoder.positional_embedding"] ?? .zeros([spec.textCtx, spec.textState]))
        self.decoderBlocks = (0..<spec.textLayers).map { i in
            WhisperMlxBlock(prefix: "decoder.blocks.\(i).", w: w, state: spec.textState, heads: spec.textHeads, cross: true, bits: spec.quantBits, groupSize: spec.quantGroupSize)
        }
        let dl = Qwen3MlxLayerNorm(dimensions: spec.textState)
        if let nw = w["decoder.ln.weight"] { dl.weight = nw }
        if let nb = w["decoder.ln.bias"] { dl.bias = nb }
        self.decoderLN = dl
        super.init()
    }

    // MARK: Encoder

    /// mel [1, 3000, nMels] -> [1, audioCtx, audioState]
    func encode(_ mel: MLXArray) -> MLXArray {
        var x = conv1d(mel, self.conv1Weight, stride: 1, padding: 1)
        if let b = self.conv1Bias {
            x = x + b.reshaped([1, 1, -1])
        }
        x = gelu(x)
        x = conv1d(x, self.conv2Weight, stride: 2, padding: 1)
        if let b = self.conv2Bias {
            x = x + b.reshaped([1, 1, -1])
        }
        x = gelu(x)
        x = x + self.positionalEmbedding
        for block in encoderBlocks {
            let (y, _) = block(x, xa: nil, mask: nil, kv: (nil, nil))
            x = y
        }
        return self.lnPost(x)
    }

    // MARK: Decoder

    /// Decode step(s). tokens [1, T]; xa encoder output.
    /// kvCache: per-layer (selfKV, crossKV) arrays.
    func decode(
        _ tokens: MLXArray, xa: MLXArray,
        kvCache: inout [((MLXArray, MLXArray)?, (MLXArray, MLXArray)?)],
        offset: Int
    ) -> (logits: MLXArray, newCache: [((MLXArray, MLXArray)?, (MLXArray, MLXArray)?)]) {
        let t = tokens.shape[1]
        var h = tokenEmbedding(tokens)
        let pe = decoderPE[offset..<(offset + t), 0...]
        h = h + pe

        if kvCache.isEmpty {
            kvCache = Array(repeating: (nil, nil), count: decoderBlocks.count)
        }

        // causal mask for the current window: [1, 1, t, offset+t]
        let mask = WhisperMlxModel.causalMask(n: t, offset: offset).asType(h.dtype)

        var cache = kvCache
        for (i, block) in decoderBlocks.enumerated() {
            let (y, c) = block(h, xa: xa, mask: mask, kv: cache[i])
            h = y
            cache[i] = c
        }
        h = self.decoderLN(h)
        // Dequantize the (possibly quantized) embedding once and reuse across
        // decode steps; the tied lm head also uses it.
        let emb: MLXArray
        if let cached = self.cachedEmbedding {
            emb = cached
        } else {
            emb = self.tokenEmbedding.effectiveWeight()
            self.cachedEmbedding = emb
        }
        let logits = matmul(h, emb.T)
        return (logits, cache)
    }

    static func causalMask(n: Int, offset: Int = 0) -> MLXArray {
        let rInds = MLXArray(0..<(offset + n)).asType(.float32)
        let lInds = offset > 0 ? MLXArray(offset..<(offset + n)).asType(.float32) : rInds
        let base = (lInds[0..., .newAxis] .< rInds[.newAxis, 0...]) * -1e9
        return base.reshaped([1, 1, n, offset + n])
    }
}
