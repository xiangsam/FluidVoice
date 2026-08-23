import Foundation
import MLX
import MLXFast
import MLXNN

// Nemotron 3.5 ASR (streaming 0.6B) MLX inference.
// Port of mlx_audio.stt.models.nemotron_asr: cache-aware causal FastConformer
// encoder + prompt kernel + RNN-T predict/joint (reuses ParakeetMlx* pieces).

// MARK: - Causal DW striding subsampling (CausalConv2D, asymmetric pad 2/1)

nonisolated final class NemotronMlxCausalDwSubsampling: Module {
    let samplingLayers: Int
    let convChannels: Int
    let out: ParakeetMlxLinear

    var convWeights: [MLXArray] = []
    var convBiases: [MLXArray?] = []

    init(spec: NemotronMlxSpec, w: [String: MLXArray]) {
        self.samplingLayers = spec.subsamplingLayers
        self.convChannels = spec.subsamplingConvChannels
        let outW = w["encoder.pre_encode.out.weight"] ?? .zeros([spec.dModel, spec.subsamplingOutputDim])
        self.out = ParakeetMlxLinear(weight: outW, bias: w["encoder.pre_encode.out.bias"])

        // NeMo conv indices (ReLU at 1/4/7): 0, 2, 3, 5, 6
        for idx in [0, 2, 3, 5, 6] {
            if let wk = w["encoder.pre_encode.conv.\(idx).weight"] {
                self.convWeights.append(wk)
                self.convBiases.append(w["encoder.pre_encode.conv.\(idx).bias"])
            }
        }
        super.init()
    }

    func subsampledLength(_ length: Int) -> Int {
        var l = length
        for _ in 0..<self.samplingLayers {
            l = (l + 2 + 1 - 3) / 2 + 1
        }
        return l
    }

    /// x [1, T, F], lengths [1] -> ([1, Tout, dModel], [1] int32)
    func callAsFunction(_ x: MLXArray, lengths: MLXArray) -> (MLXArray, MLXArray) {
        var outLengths = MLXArray([Int32(self.subsampledLength(Int(lengths[0].item(Int32.self))))])
        var y = x.expandedDimensions(axis: 3) // [1, T, F, 1]
        var idx = 0
        // layer 0: 1->C
        var p = padded(y, widths: [IntOrPair((0, 0)), IntOrPair((2, 1)), IntOrPair((2, 1)), IntOrPair((0, 0))])
        var z = conv2d(p, self.convWeights[0], stride: 2, padding: 0)
        y = maximum(z + (self.convBiases[0] ?? .zeros([self.convChannels])), 0)
        idx = 1
        for _ in 1..<self.samplingLayers {
            let dw = self.convWeights[idx]
            let pw = self.convWeights[idx + 1]
            p = padded(y, widths: [IntOrPair((0, 0)), IntOrPair((2, 1)), IntOrPair((2, 1)), IntOrPair((0, 0))])
            z = conv2d(p, dw, stride: 2, padding: 0, groups: self.convChannels)
            z = z + (self.convBiases[idx] ?? .zeros([self.convChannels]))
            z = conv2d(z, pw, stride: 1, padding: 0)
            y = maximum(z + (self.convBiases[idx + 1] ?? .zeros([self.convChannels])), 0)
            idx += 2
        }
        let t = y.shape[1]
        let f = y.shape[2]
        y = y.transposed(0, 1, 3, 2).reshaped([1, t, self.convChannels * f])
        return (self.out(y), outLengths)
    }
}

// MARK: - Rel-position attention (maskable, reuse Parakeet semantics but
// with mask support for the offline path; streaming path passes nil).

nonisolated final class NemotronMlxRelPosAttention: Module {
    let nHeads: Int
    let headDim: Int
    let scale: Float

    let linearQ: ParakeetMlxLinear
    let linearK: ParakeetMlxLinear
    let linearV: ParakeetMlxLinear
    let linearOut: ParakeetMlxLinear
    let linearPos: ParakeetMlxLinear
    var posBiasU: MLXArray
    var posBiasV: MLXArray

    init(spec: NemotronMlxSpec, prefix: String, w: [String: MLXArray]) {
        self.nHeads = spec.nHeads
        self.headDim = spec.dModel / spec.nHeads
        self.scale = powf(Float(self.headDim), -0.5)
        let d = spec.dModel
        self.linearQ = ParakeetMlxLinear(weight: w["\(prefix)linear_q.weight"] ?? .zeros([d, d]), bias: nil)
        self.linearK = ParakeetMlxLinear(weight: w["\(prefix)linear_k.weight"] ?? .zeros([d, d]), bias: nil)
        self.linearV = ParakeetMlxLinear(weight: w["\(prefix)linear_v.weight"] ?? .zeros([d, d]), bias: nil)
        self.linearOut = ParakeetMlxLinear(weight: w["\(prefix)linear_out.weight"] ?? .zeros([d, d]), bias: nil)
        self.linearPos = ParakeetMlxLinear(weight: w["\(prefix)linear_pos.weight"] ?? .zeros([d, d]), bias: nil)
        if let u = w["\(prefix)pos_bias_u"] {
            self.posBiasU = u
            self.posBiasV = w["\(prefix)pos_bias_v"] ?? .zeros(u.shape)
        } else {
            self.posBiasU = .zeros([self.nHeads, self.headDim])
            self.posBiasV = .zeros([self.nHeads, self.headDim])
        }
        super.init()
    }

    func relShift(_ x: MLXArray) -> MLXArray {
        let b = x.shape[0], h = x.shape[1], t = x.shape[2], p = x.shape[3]
        let paddedX = MLX.padded(x, widths: [IntOrPair((0, 0)), IntOrPair((0, 0)), IntOrPair((0, 0)), IntOrPair((1, 0))])
        let r = paddedX.reshaped([b, h, p + 1, t])
        let sliced = r[0..., 0..., 1..., 0...]
        return sliced.reshaped([b, h, t, p])
    }

    /// Standard attention (x = q,k,v all same); mask is additive or nil.
    func callAsFunction(_ x: MLXArray, posEmb: MLXArray, mask: MLXArray?) -> MLXArray {
        let q = self.linearQ(x)
        let k = self.linearK(x)
        let v = self.linearV(x)
        let p = self.linearPos(posEmb)

        let b = q.shape[0]
        let t = q.shape[1]
        let posLen = p.shape[1]

        let qr = q.reshaped([b, t, self.nHeads, self.headDim])
        let qu = (qr + self.posBiasU).transposed(0, 2, 1, 3)
        let qv = (qr + self.posBiasV).transposed(0, 2, 1, 3)
        let kr = k.reshaped([b, t, self.nHeads, self.headDim]).transposed(0, 2, 1, 3)
        let vr = v.reshaped([b, t, self.nHeads, self.headDim]).transposed(0, 2, 1, 3)
        let pr = p.reshaped([p.shape[0], posLen, self.nHeads, self.headDim]).transposed(0, 2, 1, 3)

        var matrixBd = matmul(qv, pr.transposed(0, 1, 3, 2))
        matrixBd = self.relShift(matrixBd)
        let kLen = kr.shape[2]
        matrixBd = matrixBd[0..., 0..., 0..., 0..<kLen] * self.scale
        if let mask {
            matrixBd = matrixBd + mask
        }

        let out = MLXFast.scaledDotProductAttention(
            queries: qu, keys: kr, values: vr, scale: self.scale, mask: matrixBd)
        let merged = out.transposed(0, 2, 1, 3).reshaped([b, t, self.nHeads * self.headDim])
        return self.linearOut(merged)
    }

    /// Cache-aware: q attends to kv window (no mask); posEmb for kv length.
    func stream(
        _ qIn: MLXArray, _ kvIn: MLXArray, posEmb: MLXArray
    ) -> MLXArray {
        let q = self.linearQ(qIn)
        let k = self.linearK(kvIn)
        let v = self.linearV(kvIn)
        let p = self.linearPos(posEmb)

        let b = q.shape[0]
        let c = q.shape[1]
        let ksz = kvIn.shape[1]
        let posLen = p.shape[1]

        let qr = q.reshaped([b, c, self.nHeads, self.headDim])
        let qu = (qr + self.posBiasU).transposed(0, 2, 1, 3)
        let qv = (qr + self.posBiasV).transposed(0, 2, 1, 3)
        let kr = k.reshaped([b, ksz, self.nHeads, self.headDim]).transposed(0, 2, 1, 3)
        let vr = v.reshaped([b, ksz, self.nHeads, self.headDim]).transposed(0, 2, 1, 3)
        let pr = p.reshaped([p.shape[0], posLen, self.nHeads, self.headDim]).transposed(0, 2, 1, 3)

        var matrixBd = matmul(qv, pr.transposed(0, 1, 3, 2))
        matrixBd = self.relShift(matrixBd)
        matrixBd = matrixBd[0..., 0..., 0..., 0..<ksz] * self.scale

        let out = MLXFast.scaledDotProductAttention(
            queries: qu, keys: kr, values: vr, scale: self.scale, mask: matrixBd)
        let merged = out.transposed(0, 2, 1, 3).reshaped([b, c, self.nHeads * self.headDim])
        return self.linearOut(merged)
    }
}

// MARK: - Conformer block (LayerNorm conv, causal depthwise conv)

nonisolated final class NemotronMlxConvolution: Module {
    let kernelSize: Int
    let padLeft: Int

    let pointwiseConv1: parakeetConv1d
    let depthwiseConv: parakeetConv1d
    let norm: ParakeetMlxLayerNorm // NeMo name: batch_norm (LayerNorm impl)
    let pointwiseConv2: parakeetConv1d

    init(prefix: String, w: [String: MLXArray], spec: NemotronMlxSpec) {
        let d = spec.dModel
        self.kernelSize = spec.convKernelSize
        self.padLeft = spec.convKernelSize - 1
        self.pointwiseConv1 = parakeetConv1d(
            weight: w["\(prefix)pointwise_conv1.weight"] ?? .zeros([d * 2, 1, d]),
            bias: spec.useBias ? w["\(prefix)pointwise_conv1.bias"] : nil)
        self.depthwiseConv = parakeetConv1d(
            weight: w["\(prefix)depthwise_conv.weight"] ?? .zeros([d, spec.convKernelSize, 1]),
            bias: spec.useBias ? w["\(prefix)depthwise_conv.bias"] : nil, groups: d)
        self.pointwiseConv2 = parakeetConv1d(
            weight: w["\(prefix)pointwise_conv2.weight"] ?? .zeros([d, 1, d]),
            bias: spec.useBias ? w["\(prefix)pointwise_conv2.bias"] : nil)
        let ln = ParakeetMlxLayerNorm(dimensions: d)
        if let ww = w["\(prefix)batch_norm.weight"] { ln.weight = ww }
        if let bb = w["\(prefix)batch_norm.bias"] { ln.bias = bb }
        self.norm = ln
        super.init()
    }

    /// Full (non-streaming) causal conv with left padding.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = self.pointwiseConv1(x) // [B, T, 2D]
        y = glu(y, axis: -1)
        y = MLX.padded(y, widths: [IntOrPair((0, 0)), IntOrPair((self.padLeft, 0)), IntOrPair((0, 0))])
        y = self.depthwiseConv(y)
        y = self.norm(y)
        y = silu(y)
        y = self.pointwiseConv2(y)
        return y
    }
}

nonisolated final class NemotronMlxFeedForward: Module {
    let linear1: ParakeetMlxLinear
    let linear2: ParakeetMlxLinear

    init(prefix: String, w: [String: MLXArray], dModel: Int, dFF: Int, useBias: Bool) {
        self.linear1 = ParakeetMlxLinear(
            weight: w["\(prefix)linear1.weight"] ?? .zeros([dFF, dModel]),
            bias: useBias ? w["\(prefix)linear1.bias"] : nil)
        self.linear2 = ParakeetMlxLinear(
            weight: w["\(prefix)linear2.weight"] ?? .zeros([dModel, dFF]),
            bias: useBias ? w["\(prefix)linear2.bias"] : nil)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        self.linear2(silu(self.linear1(x)))
    }
}

nonisolated final class NemotronMlxConformerBlock: Module {
    let ff1: NemotronMlxFeedForward
    let normFF1: ParakeetMlxLayerNorm
    let attn: NemotronMlxRelPosAttention
    let normSelfAtt: ParakeetMlxLayerNorm
    let conv: NemotronMlxConvolution
    let normConv: ParakeetMlxLayerNorm
    let ff2: NemotronMlxFeedForward
    let normFF2: ParakeetMlxLayerNorm
    let normOut: ParakeetMlxLayerNorm

    init(prefix: String, w: [String: MLXArray], spec: NemotronMlxSpec) {
        let d = spec.dModel
        let dFF = d * spec.ffExpansionFactor
        self.ff1 = NemotronMlxFeedForward(prefix: "\(prefix)feed_forward1.", w: w, dModel: d, dFF: dFF, useBias: spec.useBias)
        self.ff2 = NemotronMlxFeedForward(prefix: "\(prefix)feed_forward2.", w: w, dModel: d, dFF: dFF, useBias: spec.useBias)
        let n1 = ParakeetMlxLayerNorm(dimensions: d)
        if let ww = w["\(prefix)norm_feed_forward1.weight"] { n1.weight = ww }
        if let bb = w["\(prefix)norm_feed_forward1.bias"] { n1.bias = bb }
        self.normFF1 = n1
        let n2 = ParakeetMlxLayerNorm(dimensions: d)
        if let ww = w["\(prefix)norm_self_att.weight"] { n2.weight = ww }
        if let bb = w["\(prefix)norm_self_att.bias"] { n2.bias = bb }
        self.normSelfAtt = n2
        self.attn = NemotronMlxRelPosAttention(spec: spec, prefix: "\(prefix)self_attn.", w: w)
        let n3 = ParakeetMlxLayerNorm(dimensions: d)
        if let ww = w["\(prefix)norm_conv.weight"] { n3.weight = ww }
        if let bb = w["\(prefix)norm_conv.bias"] { n3.bias = bb }
        self.normConv = n3
        self.conv = NemotronMlxConvolution(prefix: "\(prefix)conv.", w: w, spec: spec)
        let n4 = ParakeetMlxLayerNorm(dimensions: d)
        if let ww = w["\(prefix)norm_feed_forward2.weight"] { n4.weight = ww }
        if let bb = w["\(prefix)norm_feed_forward2.bias"] { n4.bias = bb }
        self.normFF2 = n4
        let n5 = ParakeetMlxLayerNorm(dimensions: d)
        if let ww = w["\(prefix)norm_out.weight"] { n5.weight = ww }
        if let bb = w["\(prefix)norm_out.bias"] { n5.bias = bb }
        self.normOut = n5
        super.init()
    }

    func callAsFunction(_ x: MLXArray, posEmb: MLXArray, mask: MLXArray?) -> MLXArray {
        var h = x + 0.5 * self.ff1(self.normFF1(x))
        h = h + self.attn(self.normSelfAtt(h), posEmb: posEmb, mask: mask)
        h = h + self.conv(self.normConv(h))
        h = h + 0.5 * self.ff2(self.normFF2(h))
        return self.normOut(h)
    }

    /// Streaming step (per-chunk).
    func stream(_ x: MLXArray, attnCache: MLXArray?, convCache: MLXArray?, posEmb: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        let residual1 = x + 0.5 * self.ff1(self.normFF1(x))
        let xn = self.normSelfAtt(residual1)
        let kv = attnCache == nil ? xn : concatenated([attnCache!, xn], axis: 1)
        let attnOut = self.attn.stream(xn, kv, posEmb: posEmb)
        var h = residual1 + attnOut

        let xc = self.normConv(h)
        let g = glu(self.conv.pointwiseConv1(xc), axis: -1)
        let convLeft = self.conv.padLeft
        let din: MLXArray
        let convCacheVal: MLXArray
        if let convCache {
            din = concatenated([convCache, g], axis: 1)
        } else {
            din = concatenated([zeros([g.shape[0], convLeft, g.shape[2]]).asType(g.dtype), g], axis: 1)
        }
        let dw = self.conv.depthwiseConv(din)
        let dinT = din.shape[1]
        convCacheVal = din[0..., (dinT - convLeft)..., 0...]
        var y = self.conv.norm(dw)
        y = silu(y)
        y = self.conv.pointwiseConv2(y)
        h = h + y
        h = h + 0.5 * self.ff2(self.normFF2(h))
        let out = self.normOut(h)
        let leftCache = self.attnLeftCache(kv, count: self.attnCacheCount)
        return (out, leftCache, convCacheVal)
    }

    var attnCacheCount: Int { 56 }
    func attnLeftCache(_ kv: MLXArray, count: Int) -> MLXArray {
        let t = kv.shape[1]
        if t <= count { return kv }
        return kv[0..., (t - count)..., 0...]
    }
}

// MARK: - Conformer encoder (cache-aware)

nonisolated final class NemotronMlxConformer: Module {
    let spec: NemotronMlxSpec
    let preEncode: NemotronMlxCausalDwSubsampling
    let posEnc: ParakeetMlxRelPosEncoding
    let layers: [NemotronMlxConformerBlock]

    init(spec: NemotronMlxSpec, w: [String: MLXArray]) {
        self.spec = spec
        self.preEncode = NemotronMlxCausalDwSubsampling(spec: spec, w: w)
        self.posEnc = ParakeetMlxRelPosEncoding(
            dModel: spec.dModel, maxLen: spec.posEmbMaxLen, scaleInput: false)
        self.layers = (0..<spec.nLayers).map { i in
            NemotronMlxConformerBlock(prefix: "encoder.layers.\(i).", w: w, spec: spec)
        }
        super.init()
    }

    /// Full offline pass (used for large single-shot when wanted).
    func callAsFunction(_ x: MLXArray, lengths: MLXArray?) -> (MLXArray, MLXArray) {
        let lens = lengths ?? MLXArray([Int32(x.shape[1])])
        let (y, outLengths) = self.preEncode(x, lengths: lens)
        let (scaled, posEmb) = self.posEnc(y, offset: 0)
        var h = scaled
        for layer in self.layers {
            h = layer(h, posEmb: posEmb, mask: nil)
        }
        return (h, outLengths)
    }
}

// MARK: - Cache-aware streaming state (matches ConformerStreamingState)

nonisolated final class NemotronMlxStreamingState {
    let encoder: NemotronMlxConformer
    let spec: NemotronMlxSpec
    let leftCache: Int
    let chunkFrames: Int
    let chunkMel: Int
    let convLeft: Int

    var attnCache: [MLXArray?]
    var convCache: [MLXArray?]
    var melCache: MLXArray?
    var emitted = 0
    var consumed = 0
    var pending: MLXArray?
    var closed = false

    init(encoder: NemotronMlxConformer, spec: NemotronMlxSpec) {
        self.encoder = encoder
        self.spec = spec
        self.leftCache = spec.leftContext
        self.chunkFrames = spec.chunkFrames
        self.chunkMel = spec.chunkMelFrames
        self.convLeft = spec.convKernelSize - 1
        self.attnCache = Array(repeating: nil, count: spec.nLayers)
        self.convCache = Array(repeating: nil, count: spec.nLayers)
    }

    func push(_ mel: MLXArray, final: Bool = false) -> [MLXArray] {
        precondition(!self.closed, "streaming state closed")
        self.pending = self.pending == nil ? mel : concatenated([self.pending!, mel], axis: 1)
        var outputs: [MLXArray] = []
        while let pendingNow = self.pending, pendingNow.shape[1] > 0 {
            if pendingNow.shape[1] < self.chunkMel && !final { break }
            let take = min(self.chunkMel, pendingNow.shape[1])
            let m = pendingNow[0..., 0..<take, 0...]
            self.pending = pendingNow.shape[1] > take
                ? pendingNow[0..., take..., 0...]
                : nil
            let includeBoundary = final && (self.pending?.shape[1] ?? 0) == 0
            if let encoded = self.encodeMelChunk(m, includeBoundary: includeBoundary) {
                outputs.append(encoded)
            }
        }
        if final { self.closed = true }
        return outputs
    }

    private func encodeMelChunk(_ m: MLXArray, includeBoundary: Bool) -> MLXArray? {
        let cacheLen = self.melCache?.shape[1] ?? 0
        let win: MLXArray
        if let mc = self.melCache {
            win = concatenated([mc, m], axis: 1)
        } else {
            win = m
        }
        let winLen = win.shape[1]
        let sub = self.encoder.preEncode(
            win, lengths: MLXArray([Int32(winLen)])).0

        let end = self.consumed + m.shape[1]
        let base = (self.consumed - cacheLen) / self.spec.subsamplingFactor
        var lo = self.emitted - base
        let hi = includeBoundary
            ? sub.shape[1]
            : (end / self.spec.subsamplingFactor - base)
        self.consumed = end
        self.melCache = win[0..., max(0, winLen - 16)..., 0...]

        if hi <= lo {
            self.emitted = base + max(lo, hi)
            return nil
        }
        self.emitted = base + hi
        var h = sub[0..., lo..<hi, 0...]
        for i in 0..<self.encoder.layers.count {
            let block = self.encoder.layers[i]
            // attention cache length: last leftCache frames of kv (cache++h)
            let kvLen = (self.attnCache[i]?.shape[1] ?? 0) + h.shape[1]
            let posEmb = self.encoder.posEnc.posEmbFor(length: kvLen, dtype: h.dtype)
            let (out, nextAttn, nextConv) = block.stream(
                h, attnCache: self.attnCache[i], convCache: self.convCache[i], posEmb: posEmb)
            h = out
            self.attnCache[i] = nextAttn
            self.convCache[i] = nextConv
        }
        return h
    }
}

// MARK: - Prompt kernel + full model

nonisolated final class NemotronMlxPromptKernel: Module {
    let l1: ParakeetMlxLinear
    let l2: ParakeetMlxLinear

    init(spec: NemotronMlxSpec, w: [String: MLXArray]) {
        let d = spec.dModel
        self.l1 = ParakeetMlxLinear(
            weight: w["prompt_kernel.0.weight"] ?? .zeros([spec.promptHidden, d + spec.numPrompts]),
            bias: w["prompt_kernel.0.bias"])
        self.l2 = ParakeetMlxLinear(
            weight: w["prompt_kernel.2.weight"] ?? .zeros([d, spec.promptHidden]),
            bias: w["prompt_kernel.2.bias"])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        self.l2(maximum(self.l1(x), 0))
    }
}

nonisolated final class NemotronMlxModel: Module {
    let spec: NemotronMlxSpec
    let encoder: NemotronMlxConformer
    let promptKernel: NemotronMlxPromptKernel
    let decoder: ParakeetMlxPredictNetwork
    let joint: ParakeetMlxJointNetwork

    init(spec: NemotronMlxSpec, w: [String: MLXArray]) {
        self.spec = spec
        self.encoder = NemotronMlxConformer(spec: spec, w: w)
        self.promptKernel = NemotronMlxPromptKernel(spec: spec, w: w)
        self.decoder = ParakeetMlxPredictNetwork(
            embeddingCount: spec.blankAsPad ? spec.decoderVocabSize + 1 : spec.decoderVocabSize,
            predHidden: spec.predHidden,
            predRNNLayers: spec.predRNNLayers,
            w: w)
        self.joint = ParakeetMlxJointNetwork(
            encoderHidden: spec.encoderHidden,
            predHidden: spec.predHidden,
            jointHidden: spec.jointHidden,
            outputDim: spec.jointOutputDim,
            activation: spec.jointActivation,
            w: w)
        super.init()
    }

    /// Concatenate one-hot language prompt and project back to d_model.
    func applyPrompt(_ encoded: MLXArray, language: String?) -> MLXArray {
        let idx = self.spec.promptIndex(language: language)
        let b = encoded.shape[0]
        let t = encoded.shape[1]
        var oneHot = zeros([b, t, self.spec.numPrompts]).asType(encoded.dtype)
        oneHot[0..., 0..., idx] = MLXArray(1.0)
        let x = concatenated([encoded, oneHot], axis: 2)
        return self.promptKernel(x)
    }
}
