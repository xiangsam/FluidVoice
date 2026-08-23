import Foundation
import MLX
import MLXNN

// Fun-ASR (MLT) Nano MLX inference: STFT-mel+LFR+CMVN -> SANM encoder ->
// transformer adaptor -> Qwen3-0.6B LLM (reuses Qwen3 decoder pieces).
// Ported from mlx-audio-plus `stt.models.funasr` (the converter that built
// the mlx-community Fun-ASR-MLT-Nano-2512 weights).

// MARK: - SANM pieces

nonisolated final class FunMlxPositionwiseFF: Module {
    let w1: Module
    let w2: Module

    init(prefix: String, w: [String: MLXArray], idim: Int, hidden: Int, quant: Qwen3MlxQuantInfo?) {
        self.w1 = q3AsMakeLinear("\(prefix)w_1", w: w, inDim: idim, outDim: hidden, bias: true, quant: quant)
        self.w2 = q3AsMakeLinear("\(prefix)w_2", w: w, inDim: hidden, outDim: idim, bias: true, quant: quant)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        (self.w2 as! UnaryLayer)(maximum((self.w1 as! UnaryLayer)(x), 0))
    }
}

nonisolated final class FunMlxSANMAttention: Module {
    let heads: Int
    let dK: Int
    let linearQKV: Module
    let linearOut: Module
    let fsmn: parakeetConv1d
    let leftPad: Int
    let rightPad: Int

    init(
        prefix: String, w: [String: MLXArray], nHead: Int, inFeat: Int, nFeat: Int,
        kernel: Int, sanmShift: Int, quant: Qwen3MlxQuantInfo?
    ) {
        self.heads = nHead
        self.dK = nFeat / nHead
        self.linearQKV = q3AsMakeLinear("\(prefix)linear_q_k_v", w: w, inDim: inFeat, outDim: nFeat * 3, bias: true, quant: quant)
        self.linearOut = q3AsMakeLinear("\(prefix)linear_out", w: w, inDim: nFeat, outDim: nFeat, bias: true, quant: quant)
        self.fsmn = parakeetConv1d(
            weight: w["\(prefix)fsmn_block.weight"] ?? .zeros([nFeat, kernel, 1]),
            bias: nil, groups: nFeat)
        var lp = (kernel - 1) / 2
        if sanmShift > 0 { lp += sanmShift }
        self.leftPad = lp
        self.rightPad = kernel - 1 - lp
        super.init()
    }

    func fsmnForward(_ inputs: MLXArray, inputMask: MLXArray?) -> MLXArray {
        var x = inputs
        if let inputMask {
            x = x * inputMask.transposed(0, 2, 1)
        }
        x = MLX.padded(x, widths: [IntOrPair((0, 0)), IntOrPair((self.leftPad, self.rightPad)), IntOrPair((0, 0))])
        x = self.fsmn(x)
        x = x + inputs
        if let inputMask {
            x = x * inputMask.transposed(0, 2, 1)
        }
        return x
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        let b = x.shape[0]
        let t = x.shape[1]
        let qkv = (self.linearQKV as! UnaryLayer)(x)
        // split into q, k, v
        let parts = qkv.reshaped([b, t, 3, self.heads * self.dK])
        let q = parts[0..., 0..., 0, 0...]
        let k = parts[0..., 0..., 1, 0...]
        let v = parts[0..., 0..., 2, 0...]

        let fsmnMemory = self.fsmnForward(v, inputMask: mask)

        let qh = q.reshaped([b, t, self.heads, self.dK]).transposed(0, 2, 1, 3)
        let kh = k.reshaped([b, t, self.heads, self.dK]).transposed(0, 2, 1, 3)
        let vh = v.reshaped([b, t, self.heads, self.dK]).transposed(0, 2, 1, 3)

        let scores = matmul(qh * powf(Float(self.dK), -0.5), kh.transposed(0, 1, 3, 2))
        var maskedScores = scores
        if let mask {
            // mask: [B, 1, T] -> [B, 1, 1, T] (keys to block)
            let attnMask = equal(mask.reshaped([mask.shape[0], 1, 1, mask.shape[2]]), 0)
            maskedScores = which(attnMask, MLXArray(-1e9), scores)
        }
        let attn = softMax(maskedScores, axis: -1)
        let attOut = matmul(attn, vh)
        let merged = attOut.transposed(0, 2, 1, 3).reshaped([b, t, self.heads * self.dK])
        return (self.linearOut as! UnaryLayer)(merged) + fsmnMemory
    }
}

nonisolated final class FunMlxSANMLayer: Module {
    let attn: FunMlxSANMAttention
    let ff: FunMlxPositionwiseFF
    let norm1: Qwen3MlxLayerNorm
    let norm2: Qwen3MlxLayerNorm
    let inSize: Int
    let size: Int

    init(
        prefix: String, w: [String: MLXArray], inSize: Int, size: Int,
        attnHeads: Int, kernel: Int, sanmShift: Int, linearUnits: Int,
        quant: Qwen3MlxQuantInfo?
    ) {
        self.inSize = inSize
        self.size = size
        self.attn = FunMlxSANMAttention(
            prefix: "\(prefix)self_attn.", w: w, nHead: attnHeads, inFeat: inSize, nFeat: size,
            kernel: kernel, sanmShift: sanmShift, quant: quant)
        self.ff = FunMlxPositionwiseFF(
            prefix: "\(prefix)feed_forward.", w: w, idim: size, hidden: linearUnits, quant: quant)
        let n1 = Qwen3MlxLayerNorm(dimensions: inSize)
        if let nw = w["\(prefix)norm1.weight"] { n1.weight = nw }
        if let nb = w["\(prefix)norm1.bias"] { n1.bias = nb }
        self.norm1 = n1
        let n2 = Qwen3MlxLayerNorm(dimensions: size)
        if let nw = w["\(prefix)norm2.weight"] { n2.weight = nw }
        if let nb = w["\(prefix)norm2.bias"] { n2.bias = nb }
        self.norm2 = n2
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        let attnOut = self.attn(self.norm1(x), mask: mask)
        var out = self.inSize == self.size ? x + attnOut : attnOut
        out = out + self.ff(self.norm2(out))
        return out
    }
}

nonisolated final class FunMlxSANMEncoder: Module {
    let outputSize: Int
    let encoders0: [FunMlxSANMLayer] // input 560 -> 512
    let encoders: [FunMlxSANMLayer]
    let afterNorm: Qwen3MlxLayerNorm
    let tpEncoders: [FunMlxSANMLayer]
    let tpNorm: Qwen3MlxLayerNorm

    init(spec: FunMlxSpec, inputSize: Int, w: [String: MLXArray], quant: Qwen3MlxQuantInfo?) {
        self.outputSize = spec.encOutputSize
        self.encoders0 = [
            FunMlxSANMLayer(
                prefix: "audio_encoder.encoders0.0.", w: w, inSize: inputSize, size: spec.encOutputSize,
                attnHeads: spec.encHeads, kernel: spec.encKernel, sanmShift: spec.encSanmShift,
                linearUnits: spec.encLinearUnits, quant: quant)
        ]
        self.encoders = (0..<(spec.encBlocks - 1)).map { i in
            FunMlxSANMLayer(
                prefix: "audio_encoder.encoders.\(i).", w: w, inSize: spec.encOutputSize, size: spec.encOutputSize,
                attnHeads: spec.encHeads, kernel: spec.encKernel, sanmShift: spec.encSanmShift,
                linearUnits: spec.encLinearUnits, quant: quant)
        }
        let an = Qwen3MlxLayerNorm(dimensions: spec.encOutputSize)
        if let nw = w["audio_encoder.after_norm.weight"] { an.weight = nw }
        if let nb = w["audio_encoder.after_norm.bias"] { an.bias = nb }
        self.afterNorm = an
        self.tpEncoders = (0..<spec.encTPBlocks).map { i in
            FunMlxSANMLayer(
                prefix: "audio_encoder.tp_encoders.\(i).", w: w, inSize: spec.encOutputSize, size: spec.encOutputSize,
                attnHeads: spec.encHeads, kernel: spec.encKernel, sanmShift: spec.encSanmShift,
                linearUnits: spec.encLinearUnits, quant: quant)
        }
        let tn = Qwen3MlxLayerNorm(dimensions: spec.encOutputSize)
        if let nw = w["audio_encoder.tp_norm.weight"] { tn.weight = nw }
        if let nb = w["audio_encoder.tp_norm.bias"] { tn.bias = nb }
        self.tpNorm = tn
        super.init()
    }

    static func sequenceMask0(lengths: Int, maxLen: Int) -> MLXArray {
        let positions = MLXArray(0..<maxLen)[.newAxis, 0...]
        let lens = MLXArray([Int32(lengths)])[.newAxis, .newAxis, 0...]
        return (positions .< lens).asType(.float32)
    }

    func sequenceMask(lengths: Int, maxLen: Int) -> MLXArray {
        let positions = MLXArray(0..<maxLen)[.newAxis, 0...]
        let lens = MLXArray([Int32(lengths)])[.newAxis, .newAxis, 0...]
        return (positions .< lens).asType(.float32) // [1, 1, maxLen]
    }

    /// feats [1, T, D] -> [1, T', 512]
    func callAsFunction(_ xsPad: MLXArray, ilens: Int) -> MLXArray {
        let maxlen = xsPad.shape[1]
        let mask = self.sequenceMask(lengths: ilens, maxLen: maxlen)
        let scaled = xsPad * sqrt(Float(self.outputSize))
        var h = scaled
        for layer in self.encoders0 { h = layer(h, mask: mask) }
        for layer in self.encoders { h = layer(h, mask: mask) }
        h = self.afterNorm(h)
        for layer in self.tpEncoders { h = layer(h, mask: mask) }
        return self.tpNorm(h)
    }
}

// MARK: - Adaptor transformer (downsample + 2 blocks)

nonisolated final class FunMlxAdaptorAttention: Module {
    let heads: Int
    let dK: Int
    let linearQ: Module
    let linearK: Module
    let linearV: Module
    let linearOut: Module
    let scale: Float

    init(prefix: String, w: [String: MLXArray], nHead: Int, nFeat: Int, quant: Qwen3MlxQuantInfo?) {
        self.heads = nHead
        self.dK = nFeat / nHead
        self.scale = powf(Float(self.dK), -0.5)
        self.linearQ = q3AsMakeLinear("\(prefix)linear_q", w: w, inDim: nFeat, outDim: nFeat, bias: true, quant: quant)
        self.linearK = q3AsMakeLinear("\(prefix)linear_k", w: w, inDim: nFeat, outDim: nFeat, bias: true, quant: quant)
        self.linearV = q3AsMakeLinear("\(prefix)linear_v", w: w, inDim: nFeat, outDim: nFeat, bias: true, quant: quant)
        self.linearOut = q3AsMakeLinear("\(prefix)linear_out", w: w, inDim: nFeat, outDim: nFeat, bias: true, quant: quant)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        let b = x.shape[0]
        let t = x.shape[1]
        let q = (self.linearQ as! UnaryLayer)(x).reshaped([b, t, self.heads, self.dK]).transposed(0, 2, 1, 3)
        let k = (self.linearK as! UnaryLayer)(x).reshaped([b, t, self.heads, self.dK]).transposed(0, 2, 1, 3)
        let v = (self.linearV as! UnaryLayer)(x).reshaped([b, t, self.heads, self.dK]).transposed(0, 2, 1, 3)
        let scores = matmul(q * self.scale, k.transposed(0, 1, 3, 2))
        var maskedScores = scores
        if let mask {
            let attnMask = equal(mask.reshaped([mask.shape[0], 1, 1, mask.shape[2]]), 0)
            maskedScores = which(attnMask, MLXArray(-1e9), scores)
        }
        let attn = softMax(maskedScores, axis: -1)
        let out = matmul(attn, v)
        let merged = out.transposed(0, 2, 1, 3).reshaped([b, t, self.heads * self.dK])
        return (self.linearOut as! UnaryLayer)(merged)
    }
}

nonisolated final class FunMlxAdaptorLayer: Module {
    let selfAttn: FunMlxAdaptorAttention
    let ff: FunMlxPositionwiseFF
    let norm1: Qwen3MlxLayerNorm
    let norm2: Qwen3MlxLayerNorm

    init(prefix: String, w: [String: MLXArray], size: Int, heads: Int, quant: Qwen3MlxQuantInfo?) {
        self.selfAttn = FunMlxAdaptorAttention(prefix: "\(prefix)self_attn.", w: w, nHead: heads, nFeat: size, quant: quant)
        self.ff = FunMlxPositionwiseFF(prefix: "\(prefix)feed_forward.", w: w, idim: size, hidden: 256, quant: quant)
        let n1 = Qwen3MlxLayerNorm(dimensions: size)
        if let nw = w["\(prefix)norm1.weight"] { n1.weight = nw }
        if let nb = w["\(prefix)norm1.bias"] { n1.bias = nb }
        self.norm1 = n1
        let n2 = Qwen3MlxLayerNorm(dimensions: size)
        if let nw = w["\(prefix)norm2.weight"] { n2.weight = nw }
        if let nb = w["\(prefix)norm2.bias"] { n2.bias = nb }
        self.norm2 = n2
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        var h = x
        h = h + self.selfAttn(self.norm1(x), mask: mask)
        h = h + self.ff(self.norm2(h))
        return h
    }
}

nonisolated final class FunMlxAdaptor: Module {
    let linear1: Module
    let linear2: Module
    let blocks: [FunMlxAdaptorLayer]
    let dModel: Int
    let dDim: Int

    init(spec: FunMlxSpec, w: [String: MLXArray], quant: Qwen3MlxQuantInfo?) {
        self.dModel = spec.adaptorEncoderDim
        self.dDim = spec.adaptorLLMDim
        self.linear1 = q3AsMakeLinear("audio_adaptor.linear1", w: w, inDim: spec.adaptorEncoderDim, outDim: spec.adaptorFFNDim, bias: true, quant: quant)
        self.linear2 = q3AsMakeLinear("audio_adaptor.linear2", w: w, inDim: spec.adaptorFFNDim, outDim: spec.adaptorLLMDim, bias: true, quant: quant)
        self.blocks = (0..<spec.adaptorLayers).map { i in
            FunMlxAdaptorLayer(prefix: "audio_adaptor.blocks.\(i).", w: w, size: spec.adaptorLLMDim, heads: spec.adaptorHeads, quant: quant)
        }
        super.init()
    }

    /// x [1, T, 512] -> [1, T', llm dim]
    func callAsFunction(_ x: MLXArray, ilens: Int) -> (MLXArray, Int) {
        let b = x.shape[0]
        let seqLen = x.shape[1]
        let k = 1
        let chunkNum = (seqLen - 1) / k + 1
        var reshaped = x.reshaped([b, chunkNum, self.dModel * k])
        reshaped = (self.linear2 as! UnaryLayer)(maximum((self.linear1 as! UnaryLayer)(reshaped), 0))
        let outLen = min(ilens, chunkNum)
        let mask = FunMlxSANMEncoder.sequenceMask0(lengths: outLen, maxLen: reshaped.shape[1])
        var h = reshaped
        for block in self.blocks {
            h = block(h, mask: mask)
        }
        return (h, outLen)
    }
}

// MARK: - Full model

nonisolated final class FunMlxModel: Module {
    let spec: FunMlxSpec
    let encoder: FunMlxSANMEncoder
    let adaptor: FunMlxAdaptor
    let llm: Qwen3MlxTextDecoder

    init(spec: FunMlxSpec, w: [String: MLXArray], quant: Qwen3MlxQuantInfo?) {
        self.spec = spec
        self.encoder = FunMlxSANMEncoder(spec: spec, inputSize: spec.nMel * spec.lfrM, w: w, quant: quant)
        self.adaptor = FunMlxAdaptor(spec: spec, w: w, quant: quant)

        // Map llm.* keys to Qwen3 decoder naming.
        var w2: [String: MLXArray] = [:]
        for (k, v) in w {
            if k.hasPrefix("llm.model.") {
                w2[String(k.dropFirst("llm.model.".count))] = v
            }
        }

        let qspec = Qwen3MlxSpec(
            encDModel: 0, encLayers: 0, encHeads: 0, encFFNDim: 0,
            numMelBins: spec.nMel, maxSourcePositions: 1500, encOutputDim: spec.hiddenSize,
            nWindow: 50, nWindowInfer: 800, convChunkSize: 0, downsampleHidden: 0,
            hiddenSize: spec.hiddenSize,
            numLayers: spec.numLayers,
            numHeads: spec.numHeads,
            numKVHeads: spec.numKVHeads,
            headDim: spec.headDim,
            intermediateSize: spec.intermediateSize,
            vocabSize: spec.vocabSize,
            rmsEps: spec.rmsEps,
            ropeTheta: spec.ropeTheta)
        let decoder = Qwen3MlxTextDecoder(spec: qspec, w: w2, quant: quant)
        // Untied LM head: llm.lm_head.weight exists in the converted checkpoint
        // (config `tie_word_embeddings: false`). Using the tied embedding as the
        // projection yields constant/degenerate output - load the real head.
        if let lh = w["llm.lm_head.weight"] {
            // Keep the head in fp16: the float32 math path only needs fp32
            // activations (cast at the input), fp16 weights avoid holding an
            // extra ~620 MB copy of the vocab-projection weights.
            decoder.lmHeadWeight = lh
            if let sc = w["llm.lm_head.scales"] {
                decoder.lmHeadScales = sc
                decoder.lmHeadBiases = w["llm.lm_head.biases"]
                decoder.lmHeadGroupSize = quant?.groupSize ?? 64
                decoder.lmHeadBits = quant?.bits ?? 8
            }
        }
        self.llm = decoder
        super.init()
    }
}
