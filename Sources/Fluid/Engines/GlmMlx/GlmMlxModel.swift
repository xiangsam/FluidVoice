import Foundation
import MLX
import MLXFast
import MLXNN

// GLM-ASR-Nano MLX inference: Whisper-style audio encoder (Conv1d + RoPE)
// + MLP adapter (merge x4) + LLaMA decoder. Reuses injectable layers and
// the quantized-linear/embedding helpers built for Qwen3/Parakeet.

// MARK: - Whisper audio encoder

nonisolated final class GlmMlxWhisperAttention: Module {
    let heads: Int
    let headDim: Int
    let scale: Float
    let qProj: Module
    let kProj: Module
    let vProj: Module
    let outProj: Module
    let rope: RoPE

    init(spec: GlmMlxSpec, prefix: String, w: [String: MLXArray], quant: Qwen3MlxQuantInfo?) {
        self.heads = spec.encHeads
        self.headDim = spec.encDModel / spec.encHeads
        self.scale = powf(Float(self.headDim), -0.5)
        let d = spec.encDModel
        self.qProj = q3AsMakeLinear("\(prefix)q_proj", w: w, inDim: d, outDim: d, bias: true, quant: quant)
        self.kProj = q3AsMakeLinear("\(prefix)k_proj", w: w, inDim: d, outDim: d, bias: false, quant: quant)
        self.vProj = q3AsMakeLinear("\(prefix)v_proj", w: w, inDim: d, outDim: d, bias: true, quant: quant)
        self.outProj = q3AsMakeLinear("\(prefix)out_proj", w: w, inDim: d, outDim: d, bias: true, quant: quant)
        self.rope = RoPE(dimensions: spec.ropeDim, traditional: spec.ropeTraditional, base: 10_000)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let b = x.shape[0]
        let t = x.shape[1]
        let h = self.heads
        let d = self.headDim

        var q = (self.qProj as! UnaryLayer)(x).reshaped([b, t, h, d])
        let k = (self.kProj as! UnaryLayer)(x).reshaped([b, t, h, d])
        let v = (self.vProj as! UnaryLayer)(x).reshaped([b, t, h, d])

        q = q.transposed(0, 2, 1, 3)
        let kT = k.transposed(0, 2, 1, 3)
        let vT = v.transposed(0, 2, 1, 3)

        q = self.rope(q, offset: 0)
        let kR = self.rope(kT, offset: 0)

        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: kR, values: vT, scale: self.scale, mask: nil)
        let merged = out.transposed(0, 2, 1, 3).reshaped([b, t, h * d])
        return (self.outProj as! UnaryLayer)(merged)
    }
}

nonisolated final class GlmMlxWhisperLayer: Module {
    let selfAttnNorm: Qwen3MlxLayerNorm
    let selfAttn: GlmMlxWhisperAttention
    let finalNorm: Qwen3MlxLayerNorm
    let fc1: Module
    let fc2: Module

    init(spec: GlmMlxSpec, prefix: String, w: [String: MLXArray], quant: Qwen3MlxQuantInfo?) {
        let d = spec.encDModel
        let n1 = Qwen3MlxLayerNorm(dimensions: d)
        if let nw = w["\(prefix)self_attn_layer_norm.weight"] { n1.weight = nw }
        if let nb = w["\(prefix)self_attn_layer_norm.bias"] { n1.bias = nb }
        self.selfAttnNorm = n1
        self.selfAttn = GlmMlxWhisperAttention(spec: spec, prefix: "\(prefix)self_attn.", w: w, quant: quant)
        let n2 = Qwen3MlxLayerNorm(dimensions: d)
        if let nw = w["\(prefix)final_layer_norm.weight"] { n2.weight = nw }
        if let nb = w["\(prefix)final_layer_norm.bias"] { n2.bias = nb }
        self.finalNorm = n2
        self.fc1 = q3AsMakeLinear("\(prefix)fc1", w: w, inDim: d, outDim: spec.encFFNDim, bias: true, quant: quant)
        self.fc2 = q3AsMakeLinear("\(prefix)fc2", w: w, inDim: spec.encFFNDim, outDim: d, bias: true, quant: quant)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x + self.selfAttn(self.selfAttnNorm(x))
        h = h + (self.fc2 as! UnaryLayer)(gelu((self.fc1 as! UnaryLayer)(self.finalNorm(h))))
        return h
    }
}

nonisolated final class GlmMlxWhisperEncoder: Module {
    let conv1: parakeetConv1d
    let conv2: parakeetConv1d
    let layers: [GlmMlxWhisperLayer]

    init(spec: GlmMlxSpec, w: [String: MLXArray], quant: Qwen3MlxQuantInfo?) {
        let d = spec.encDModel
        self.conv1 = parakeetConv1d(
            weight: w["audio_encoder.whisper.conv1.weight"] ?? .zeros([d, 3, spec.numMelBins]),
            bias: w["audio_encoder.whisper.conv1.bias"])
        self.conv2 = parakeetConv1d(
            weight: w["audio_encoder.whisper.conv2.weight"] ?? .zeros([d, 3, d]),
            bias: w["audio_encoder.whisper.conv2.bias"])
        self.layers = (0..<spec.encLayers).map { i in
            GlmMlxWhisperLayer(spec: spec, prefix: "audio_encoder.whisper.layers.\(i).", w: w, quant: quant)
        }
        super.init()
    }

    /// x: [B, T, nMel] (channels-last) -> [B, T', d_model]
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = gelu(self.conv1(x)) // [B, T, 1280]
        h = gelu(self.conv2(h)) // stride 2 -> [B, T/2, 1280]
        for layer in self.layers {
            h = layer(h)
        }
        return h
    }
}

// MARK: - Adapter

nonisolated final class GlmMlxAdapter: Module {
    let fc1: Module
    let fc2: Module

    init(spec: GlmMlxSpec, w: [String: MLXArray], quant: Qwen3MlxQuantInfo?) {
        // merged_dim = d_model * merge_factor
        self.fc1 = q3AsMakeLinear(
            "audio_encoder.adapting.fc1", w: w,
            inDim: spec.encDModel * spec.mergeFactor,
            outDim: spec.hiddenSize * 2, bias: true, quant: quant)
        self.fc2 = q3AsMakeLinear(
            "audio_encoder.adapting.fc2", w: w,
            inDim: spec.hiddenSize * 2, outDim: spec.hiddenSize, bias: true, quant: quant)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        (self.fc2 as! UnaryLayer)(gelu((self.fc1 as! UnaryLayer)(x)))
    }
}

// MARK: - LLaMA decoder (no QK-norm)

nonisolated final class GlmMlxAttention: Module {
    let nHeads: Int
    let nKVHeads: Int
    let headDim: Int
    let scale: Float
    let qProj: Module
    let kProj: Module
    let vProj: Module
    let oProj: Module
    let rope: RoPE

    init(prefix: String, w: [String: MLXArray], spec: GlmMlxSpec, quant: Qwen3MlxQuantInfo?) {
        self.nHeads = spec.numHeads
        self.nKVHeads = spec.numKVHeads
        self.headDim = spec.headDim
        self.scale = powf(Float(spec.headDim), -0.5)
        let h = spec.hiddenSize
        self.qProj = q3AsMakeLinear("\(prefix)q_proj", w: w, inDim: h, outDim: self.nHeads * self.headDim, bias: false, quant: quant)
        self.kProj = q3AsMakeLinear("\(prefix)k_proj", w: w, inDim: h, outDim: self.nKVHeads * self.headDim, bias: false, quant: quant)
        self.vProj = q3AsMakeLinear("\(prefix)v_proj", w: w, inDim: h, outDim: self.nKVHeads * self.headDim, bias: false, quant: quant)
        self.oProj = q3AsMakeLinear("\(prefix)o_proj", w: w, inDim: self.nHeads * self.headDim, outDim: h, bias: false, quant: quant)
        self.rope = RoPE(dimensions: spec.headDim, traditional: false, base: spec.ropeTheta)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cache: Qwen3MlxKVCache?, layerIdx: Int) -> MLXArray {
        let b = x.shape[0]
        let t = x.shape[1]

        var q = (self.qProj as! UnaryLayer)(x).reshaped([b, t, self.nHeads, self.headDim])
        var k = (self.kProj as! UnaryLayer)(x).reshaped([b, t, self.nKVHeads, self.headDim])
        let v = (self.vProj as! UnaryLayer)(x).reshaped([b, t, self.nKVHeads, self.headDim])

        q = q.transposed(0, 2, 1, 3)
        k = k.transposed(0, 2, 1, 3)
        let vT = v.transposed(0, 2, 1, 3)

        let offset = cache?.offset ?? 0
        q = self.rope(q, offset: offset)
        k = self.rope(k, offset: offset)

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
            queries: q, keys: kFull, values: vFull, scale: self.scale, mask: mask)
        let merged = out.transposed(0, 2, 1, 3).reshaped([b, t, self.nHeads * self.headDim])
        return (self.oProj as! UnaryLayer)(merged)
    }
}

nonisolated final class GlmMlxMLP: Module {
    let gateProj: Module
    let upProj: Module
    let downProj: Module

    init(prefix: String, w: [String: MLXArray], spec: GlmMlxSpec, quant: Qwen3MlxQuantInfo?) {
        let h = spec.hiddenSize
        self.gateProj = q3AsMakeLinear("\(prefix)gate_proj", w: w, inDim: h, outDim: spec.intermediateSize, bias: false, quant: quant)
        self.upProj = q3AsMakeLinear("\(prefix)up_proj", w: w, inDim: h, outDim: spec.intermediateSize, bias: false, quant: quant)
        self.downProj = q3AsMakeLinear("\(prefix)down_proj", w: w, inDim: spec.intermediateSize, outDim: h, bias: false, quant: quant)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let g = silu((self.gateProj as! UnaryLayer)(x))
        let u = (self.upProj as! UnaryLayer)(x)
        return (self.downProj as! UnaryLayer)(g * u)
    }
}

nonisolated final class GlmMlxDecoderLayer: Module {
    let inputLayerNorm: Qwen3MlxRMSNorm
    let selfAttn: GlmMlxAttention
    let postAttentionLayerNorm: Qwen3MlxRMSNorm
    let mlp: GlmMlxMLP

    init(prefix: String, w: [String: MLXArray], spec: GlmMlxSpec, quant: Qwen3MlxQuantInfo?) {
        let n1 = Qwen3MlxRMSNorm(dimensions: spec.hiddenSize, eps: spec.rmsEps)
        if let nw = w["\(prefix)input_layernorm.weight"] { n1.weight = nw }
        self.inputLayerNorm = n1
        self.selfAttn = GlmMlxAttention(prefix: "\(prefix)self_attn.", w: w, spec: spec, quant: quant)
        let n2 = Qwen3MlxRMSNorm(dimensions: spec.hiddenSize, eps: spec.rmsEps)
        if let nw = w["\(prefix)post_attention_layernorm.weight"] { n2.weight = nw }
        self.postAttentionLayerNorm = n2
        self.mlp = GlmMlxMLP(prefix: "\(prefix)mlp.", w: w, spec: spec, quant: quant)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cache: Qwen3MlxKVCache?, layerIdx: Int) -> MLXArray {
        var h = x + self.selfAttn(self.inputLayerNorm(x), cache: cache, layerIdx: layerIdx)
        h = h + self.mlp(self.postAttentionLayerNorm(h))
        return h
    }
}

// MARK: - Full model

nonisolated final class GlmMlxModel: Module {
    let spec: GlmMlxSpec
    let whisper: GlmMlxWhisperEncoder
    let encoderLayerNorm: Qwen3MlxLayerNorm
    let adapter: GlmMlxAdapter
    let audioBosEos: Qwen3MlxEmbedding
    let embedTokens: Qwen3MlxEmbedding
    let layers: [GlmMlxDecoderLayer]
    let norm: Qwen3MlxRMSNorm
    let lmHead: Qwen3MlxEmbedding // quantized [vocab, hidden]

    init(spec: GlmMlxSpec, w: [String: MLXArray], quant: Qwen3MlxQuantInfo?) {
        self.spec = spec
        self.whisper = GlmMlxWhisperEncoder(spec: spec, w: w, quant: quant)
        let eln = Qwen3MlxLayerNorm(dimensions: spec.encDModel)
        if let nw = w["audio_encoder.layer_norm.weight"] { eln.weight = nw }
        if let nb = w["audio_encoder.layer_norm.bias"] { eln.bias = nb }
        self.encoderLayerNorm = eln
        self.adapter = GlmMlxAdapter(spec: spec, w: w, quant: quant)

        let abe = Qwen3MlxEmbedding(embeddingCount: 2, dimensions: spec.hiddenSize)
        if let ew = w["audio_encoder.audio_bos_eos_token.weight"] {
            abe.weight = ew
            if let es = w["audio_encoder.audio_bos_eos_token.scales"] {
                abe.scales = es
                abe.biases = w["audio_encoder.audio_bos_eos_token.biases"]
                abe.groupSize = quant?.groupSize ?? 64
                abe.bits = quant?.bits ?? 8
                abe.quantized = true
            }
        }
        self.audioBosEos = abe

        let emb = Qwen3MlxEmbedding(embeddingCount: spec.vocabSize, dimensions: spec.hiddenSize)
        if let ew = w["model.embed_tokens.weight"] {
            emb.weight = ew
            if let es = w["model.embed_tokens.scales"] {
                emb.scales = es
                emb.biases = w["model.embed_tokens.biases"]
                emb.groupSize = quant?.groupSize ?? 64
                emb.bits = quant?.bits ?? 8
                emb.quantized = true
            }
        }
        self.embedTokens = emb
        self.layers = (0..<spec.numLayers).map { i in
            GlmMlxDecoderLayer(prefix: "model.layers.\(i).", w: w, spec: spec, quant: quant)
        }
        let n = Qwen3MlxRMSNorm(dimensions: spec.hiddenSize, eps: spec.rmsEps)
        if let nw = w["model.norm.weight"] { n.weight = nw }
        self.norm = n

        let head = Qwen3MlxEmbedding(embeddingCount: spec.vocabSize, dimensions: spec.hiddenSize)
        if let ew = w["lm_head.weight"] {
            head.weight = ew
            if let es = w["lm_head.scales"] {
                head.scales = es
                head.biases = w["lm_head.biases"]
                head.groupSize = quant?.groupSize ?? 64
                head.bits = quant?.bits ?? 8
                head.quantized = true
            }
        }
        self.lmHead = head
        super.init()
    }

    /// Encode audio -> merged/prompt-ready audio embeddings [1, T', hidden].
    func encodeAudio(_ mel: MLXArray) -> (MLXArray, Int) {
        let feats = self.whisper(mel) // [1, T, 1280]
        let normed = self.encoderLayerNorm(feats)
        let b = normed.shape[0]
        let seqLen = normed.shape[1]
        let mf = self.spec.mergeFactor
        let newSeq = (seqLen - mf) / mf + 1
        var merged: [MLXArray] = []
        for i in 0..<newSeq {
            let chunk = normed[0..., (i * mf)..<((i + 1) * mf), 0...]
                .reshaped([b, mf * self.spec.encDModel])
            merged.append(chunk)
        }
        let mergedAudio = stacked(merged, axis: 1)
        return (self.adapter(mergedAudio), newSeq)
    }

    /// LLaMA forward over embeddings.
    func llama(
        _ inputs: MLXArray?, cache: Qwen3MlxKVCache?, isEmbeds: Bool
    ) -> MLXArray {
        var h = isEmbeds ? inputs! : self.embedTokens(inputs!)
        for (i, layer) in self.layers.enumerated() {
            h = layer(h, cache: cache, layerIdx: i)
        }
        h = self.norm(h)
        return matmul(h, self.lmHead.effectiveWeight().T)
    }
}
