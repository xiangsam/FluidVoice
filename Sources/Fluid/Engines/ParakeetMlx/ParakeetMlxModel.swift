import Foundation
import MLX
import MLXFast
import MLXNN

// Parakeet TDT MLX inference port of parakeet-mlx (NVIDIA, MIT).
// Weights are injected at construction time from the safetensors table
// (all F32 layers; this family ships unquantized).

// MARK: - Small injectable layers

nonisolated final class ParakeetMlxLinear: Module {
    var weight: MLXArray
    var bias: MLXArray?

    init(weight: MLXArray, bias: MLXArray?) {
        self.weight = weight
        self.bias = bias
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let out = matmul(x, self.weight.T)
        if let bias { return out + bias }
        return out
    }
}

nonisolated final class ParakeetMlxLayerNorm: Module {
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
        MLXFast.layerNorm(x, weight: self.weight, bias: self.bias, eps: self.eps)
    }
}

nonisolated final class ParakeetMlxBatchNorm: Module {
    var weight: MLXArray
    var bias: MLXArray
    var runningMean: MLXArray
    var runningVar: MLXArray
    let eps: Float

    init(dimensions: Int, eps: Float = 1e-5) {
        self.weight = MLXArray.ones([dimensions])
        self.bias = MLXArray.zeros([dimensions])
        self.runningMean = MLXArray.zeros([dimensions])
        self.runningVar = MLXArray.ones([dimensions])
        self.eps = eps
        super.init()
    }

    /// x: [B, T, C]; normalized over last axis (channels-last events).
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let mean = self.runningMean[.newAxis, 0...]
        let var_ = self.runningVar[.newAxis, 0...]
        let w = self.weight[.newAxis, 0...]
        let b = self.bias[.newAxis, 0...]
        return (x - mean) / sqrt(var_ + self.eps) * w + b
    }
}

// MARK: - DW striding subsampling

/// mel [1, T, F] -> features [1, Tout, dModel], lengths [1] (F32).
nonisolated final class ParakeetMlxDwSubsampling: Module {
    let samplingLayers: Int
    let convChannels: Int
    let out: ParakeetMlxLinear

    // python module list order: conv(0), relu(1), dw(2), pw(3), relu(4),
    // dw(5), pw(6), relu(7)
    var convWeights: [MLXArray] = []
    var convBiases: [MLXArray] = []

    init(spec: ParakeetMlxSpec, w: [String: MLXArray]) {
        self.samplingLayers = spec.subsamplingLayers
        self.convChannels = spec.subsamplingConvChannels
        guard let outW = w["encoder.pre_encode.out.weight"] else {
            fatalError("missing encoder.pre_encode.out.weight")
        }
        self.out = ParakeetMlxLinear(weight: outW, bias: w["encoder.pre_encode.out.bias"])

        for idx in [0, 2, 3, 5, 6] {
            if let wk = w["encoder.pre_encode.conv.\(idx).weight"] {
                self.convWeights.append(wk)
                self.convBiases.append(w["encoder.pre_encode.conv.\(idx).bias"] ?? MLXArray.zeros([wk.shape[0]]))
            }
        }
        super.init()
    }

    /// x: [1, T, F], lengths: [1] -> eaten [[1, Tout, dModel], lengths]
    func callAsFunction(_ x: MLXArray, lengths: MLXArray) -> (MLXArray, MLXArray) {
        var outLengths = lengths
        for _ in 0..<self.samplingLayers {
            outLengths = ((outLengths + 2 - 3) / 2 + 1)
        }
        outLengths = outLengths.asType(.int32)

        var y = x.expandedDimensions(axis: 3) // [1, T, F, 1] = [B, H=time, W=freq, C]
        // conv.0: 1->C 3x3 stride2
        var z = conv2d(y, self.convWeights[0], stride: 2, padding: 1)
        y = maximum(z + self.convBiases[0], 0)
        // dw + pw x2
        for step in 0..<2 {
            let dwI = 1 + step * 2
            let pwI = 2 + step * 2
            z = conv2d(y, self.convWeights[dwI], stride: 2, padding: 1, groups: self.convChannels)
            y = z + self.convBiases[dwI]
            z = conv2d(y, self.convWeights[pwI], stride: 1, padding: 0)
            y = maximum(z + self.convBiases[pwI], 0)
        }
        // [1, T', F', C] -> [1, T', C * F'] (C-major)
        let t = y.shape[1]
        let f = y.shape[2]
        y = y.transposed(0, 1, 3, 2).reshaped([1, t, self.convChannels * f])
        return (self.out(y), outLengths)
    }
}

// MARK: - Relative positional encoding

nonisolated final class ParakeetMlxRelPosEncoding: Module {
    let dModel: Int
    let maxLen: Int
    let scale: Float
    var pe: MLXArray // [1, 2*maxLen-1, dModel] float32

    init(dModel: Int, maxLen: Int, scaleInput: Bool) {
        self.dModel = dModel
        self.maxLen = maxLen
        self.scale = scaleInput ? sqrt(Float(dModel)) : 1.0
        self.pe = Self.buildPE(maxLen: maxLen, dModel: dModel)
        super.init()
    }

    static func buildPE(maxLen: Int, dModel: Int) -> MLXArray {
        let count = 2 * maxLen - 1
        // python: mx.arange(max_len - 1, -max_len, -1)
        let positions = MLXArray((0..<count).map { maxLen - 1 - $0 })
            .asType(.float32).expandedDimensions(axis: 1) // [(cnt), 1]
        let divTerm = exp(
            (MLXArray((0..<(dModel / 2)).map { $0 * 2 }).asType(.float32) * -(log(10_000.0) / Float(dModel))))
        var pe = MLXArray.zeros([count, dModel])
        let sinPart = sin(positions * divTerm[.newAxis, 0...])
        let cosPart = cos(positions * divTerm[.newAxis, 0...])
        // pe[:, 0::2] = sin; pe[:, 1::2] = cos
        // build via concatenate interleave
        let interleaved = Self.interleave(a: sinPart, b: cosPart)
        pe = interleaved
        return pe.expandedDimensions(axis: 0)
    }

    static func interleave(a: MLXArray, b: MLXArray) -> MLXArray {
        // [R, C] -> [R, 2C] with a at even cols, b at odd cols
        let stacked = stacked([a, b], axis: -1) // [R, C, 2]
        return stacked.reshaped([a.shape[0], a.shape[1] * 2])
    }

    /// x [B, T, D] -> (scaled x, posEmb [1, 2*T-1, D])
    func callAsFunction(_ x: MLXArray, offset: Int = 0) -> (MLXArray, MLXArray) {
        let inputLen = x.shape[1] + offset
        let bufferLen = self.pe.shape[1]
        let startIdx = bufferLen / 2 - (inputLen - 1)
        let endIdx = bufferLen / 2 + (inputLen - 1) + 1
        let posEmb = self.pe[0..., startIdx..<endIdx, 0...].asType(x.dtype)
        return (x * self.scale, posEmb)
    }

    /// Positional embedding for a window of `length` frames (2*length-1),
    /// centered on the current position (streaming cache windows).
    func posEmbFor(length: Int, dtype: DType = .float32) -> MLXArray {
        let bufferLen = self.pe.shape[1]
        let startIdx = bufferLen / 2 - (length - 1)
        let endIdx = bufferLen / 2 + (length - 1) + 1
        return self.pe[0..., startIdx..<endIdx, 0...].asType(dtype)
    }
}

// MARK: - Relative position self-attention

nonisolated final class ParakeetMlxRelPosAttention: Module {
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

    init(spec: ParakeetMlxSpec, prefix: String, w: [String: MLXArray]) {
        self.nHeads = spec.nHeads
        self.headDim = spec.dModel / spec.nHeads
        self.scale = powf(Float(self.headDim), -0.5)
        let d = spec.dModel
        self.linearQ = ParakeetMlxLinear(weight: w["\(prefix)linear_q.weight"] ?? .zeros([d, d]), bias: w["\(prefix)linear_q.bias"])
        self.linearK = ParakeetMlxLinear(weight: w["\(prefix)linear_k.weight"] ?? .zeros([d, d]), bias: w["\(prefix)linear_k.bias"])
        self.linearV = ParakeetMlxLinear(weight: w["\(prefix)linear_v.weight"] ?? .zeros([d, d]), bias: w["\(prefix)linear_v.bias"])
        self.linearOut = ParakeetMlxLinear(weight: w["\(prefix)linear_out.weight"] ?? .zeros([d, d]), bias: w["\(prefix)linear_out.bias"])
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
        // x: [B, H, T, P] -> [B, H, T, P]
        let b = x.shape[0], h = x.shape[1], t = x.shape[2], p = x.shape[3]
        let padded = MLX.padded(x, widths: [IntOrPair((0, 0)), IntOrPair((0, 0)), IntOrPair((0, 0)), IntOrPair((1, 0))])
        let r = padded.reshaped([b, h, p + 1, t])
        let sliced = r[0..., 0..., 1..., 0...]
        return sliced.reshaped([b, h, t, p])
    }

    func callAsFunction(
        _ q: MLXArray, _ k: MLXArray, _ v: MLXArray,
        posEmb: MLXArray, mask: MLXArray?
    ) -> MLXArray {
        let q = self.linearQ(q) // [B, T, D]
        let k = self.linearK(k)
        let v = self.linearV(v)
        let p = self.linearPos(posEmb) // [1, P, D]

        let b = q.shape[0]
        let t = q.shape[1]
        let posLen = p.shape[1]

        let qr = q.reshaped([b, t, self.nHeads, self.headDim])
        let qu = (qr + self.posBiasU).transposed(0, 2, 1, 3)
        let qv = (qr + self.posBiasV).transposed(0, 2, 1, 3)
        let kr = k.reshaped([b, k.shape[1], self.nHeads, self.headDim]).transposed(0, 2, 1, 3)
        let vr = v.reshaped([b, v.shape[1], self.nHeads, self.headDim]).transposed(0, 2, 1, 3)
        let pr = p.reshaped([p.shape[0], posLen, self.nHeads, self.headDim]).transposed(0, 2, 1, 3)

        var matrixBd = matmul(qv, pr.transposed(0, 1, 3, 2)) // [1, H, T, P]
        matrixBd = self.relShift(matrixBd)
        let kLen = kr.shape[2]
        matrixBd = matrixBd[0..., 0..., 0..., 0..<kLen] * self.scale
        // (Non-streaming inference never passes a padding mask; batch always 1.)

        let out = MLXFast.scaledDotProductAttention(
            queries: qu, keys: kr, values: vr, scale: self.scale, mask: matrixBd)
        let merged = out.transposed(0, 2, 1, 3).reshaped([b, t, self.nHeads * self.headDim])
        return self.linearOut(merged)
    }
}

// MARK: - Conformer block

nonisolated final class ParakeetMlxFeedForward: Module {
    let linear1: ParakeetMlxLinear
    let linear2: ParakeetMlxLinear

    init(prefix: String, w: [String: MLXArray], dModel: Int, dFF: Int, bias: Bool, spec: ParakeetMlxSpec) {
        self.linear1 = ParakeetMlxLinear(
            weight: w["\(prefix)linear1.weight"] ?? .zeros([dFF, dModel]),
            bias: bias ? w["\(prefix)linear1.bias"] : nil)
        self.linear2 = ParakeetMlxLinear(
            weight: w["\(prefix)linear2.weight"] ?? .zeros([dModel, dFF]),
            bias: bias ? w["\(prefix)linear2.bias"] : nil)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = silu(self.linear1(x))
        return self.linear2(h)
    }
}

nonisolated final class ParakeetMlxConvolution: Module {
    let kernelSize: Int
    let padding: Int

    let pointwiseConv1: parakeetConv1d
    let depthwiseConv: parakeetConv1d
    let batchNorm: ParakeetMlxBatchNorm
    let pointwiseConv2: parakeetConv1d

    init(prefix: String, w: [String: MLXArray], spec: ParakeetMlxSpec) {
        let d = spec.dModel
        self.kernelSize = spec.convKernelSize
        self.padding = (spec.convKernelSize - 1) / 2
        self.pointwiseConv1 = parakeetConv1d(
            weight: w["\(prefix)pointwise_conv1.weight"] ?? .zeros([d * 2, 1, d]),
            bias: w["\(prefix)pointwise_conv1.bias"])
        self.depthwiseConv = parakeetConv1d(
            weight: w["\(prefix)depthwise_conv.weight"] ?? .zeros([d, spec.convKernelSize, 1]),
            bias: w["\(prefix)depthwise_conv.bias"], groups: d)
        self.pointwiseConv2 = parakeetConv1d(
            weight: w["\(prefix)pointwise_conv2.weight"] ?? .zeros([d, 1, d]),
            bias: w["\(prefix)pointwise_conv2.bias"])

        let bn = ParakeetMlxBatchNorm(dimensions: d)
        if let ww = w["\(prefix)batch_norm.weight"] { bn.weight = ww }
        if let bb = w["\(prefix)batch_norm.bias"] { bn.bias = bb }
        if let rm = w["\(prefix)batch_norm.running_mean"] { bn.runningMean = rm }
        if let rv = w["\(prefix)batch_norm.running_var"] { bn.runningVar = rv }
        self.batchNorm = bn
        super.init()
    }

    /// x: [B, T, D]
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = self.pointwiseConv1(x) // [B, T, 2D]
        y = glu(y, axis: -1) // GLU over last dim -> [B, T, D]
        y = MLX.padded(y, widths: [IntOrPair((0, 0)), IntOrPair((self.padding, self.padding)), IntOrPair((0, 0))])
        y = self.depthwiseConv(y)
        y = self.batchNorm(y)
        y = silu(y)
        y = self.pointwiseConv2(y)
        return y
    }
}

/// Conv1d wrapper: weight [O, K, C], input [B, T, C].
nonisolated final class parakeetConv1d: Module {
    let weight: MLXArray
    let bias: MLXArray?
    let groups: Int

    init(weight: MLXArray, bias: MLXArray?, groups: Int = 1) {
        self.weight = weight
        self.bias = bias
        self.groups = groups
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = conv1d(x, self.weight, stride: 1, padding: 0, groups: self.groups)
        if let bias {
            y = y + bias[.newAxis, 0...]
        }
        return y
    }
}

nonisolated final class ParakeetMlxConformerBlock: Module {
    let ff1: ParakeetMlxFeedForward
    let normFF1: ParakeetMlxLayerNorm
    let attn: ParakeetMlxRelPosAttention
    let normSelfAtt: ParakeetMlxLayerNorm
    let conv: ParakeetMlxConvolution
    let normConv: ParakeetMlxLayerNorm
    let ff2: ParakeetMlxFeedForward
    let normFF2: ParakeetMlxLayerNorm
    let normOut: ParakeetMlxLayerNorm

    init(prefix: String, w: [String: MLXArray], spec: ParakeetMlxSpec) {
        let d = spec.dModel
        let ffDim = d * spec.ffExpansionFactor
        self.ff1 = ParakeetMlxFeedForward(prefix: "\(prefix)feed_forward1.", w: w, dModel: d, dFF: ffDim, bias: spec.useBias, spec: spec)
        self.ff2 = ParakeetMlxFeedForward(prefix: "\(prefix)feed_forward2.", w: w, dModel: d, dFF: ffDim, bias: spec.useBias, spec: spec)
        let n1 = ParakeetMlxLayerNorm(dimensions: d)
        if let ww = w["\(prefix)norm_feed_forward1.weight"] { n1.weight = ww }
        if let bb = w["\(prefix)norm_feed_forward1.bias"] { n1.bias = bb }
        self.normFF1 = n1
        let n2 = ParakeetMlxLayerNorm(dimensions: d)
        if let ww = w["\(prefix)norm_self_att.weight"] { n2.weight = ww }
        if let bb = w["\(prefix)norm_self_att.bias"] { n2.bias = bb }
        self.normSelfAtt = n2
        self.attn = ParakeetMlxRelPosAttention(spec: spec, prefix: "\(prefix)self_attn.", w: w)
        let n3 = ParakeetMlxLayerNorm(dimensions: d)
        if let ww = w["\(prefix)norm_conv.weight"] { n3.weight = ww }
        if let bb = w["\(prefix)norm_conv.bias"] { n3.bias = bb }
        self.normConv = n3
        self.conv = ParakeetMlxConvolution(prefix: "\(prefix)conv.", w: w, spec: spec)
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

    func callAsFunction(_ x: MLXArray, posEmb: MLXArray) -> MLXArray {
        var h = x + 0.5 * self.ff1(self.normFF1(x))
        h = h + self.attn(self.normSelfAtt(h), self.normSelfAtt(h), self.normSelfAtt(h), posEmb: posEmb, mask: nil)
        h = h + self.conv(self.normConv(h))
        h = h + 0.5 * self.ff2(self.normFF2(h))
        return self.normOut(h)
    }
}

// MARK: - Conformer encoder

nonisolated final class ParakeetMlxConformer: Module {
    let spec: ParakeetMlxSpec
    let preEncode: ParakeetMlxDwSubsampling
    let posEnc: ParakeetMlxRelPosEncoding
    let layers: [ParakeetMlxConformerBlock]

    init(spec: ParakeetMlxSpec, w: [String: MLXArray]) {
        self.spec = spec
        self.preEncode = ParakeetMlxDwSubsampling(spec: spec, w: w)
        self.posEnc = ParakeetMlxRelPosEncoding(
            dModel: spec.dModel, maxLen: spec.posEmbMaxLen, scaleInput: spec.xscaling)
        self.layers = (0..<spec.nLayers).map { i in
            ParakeetMlxConformerBlock(prefix: "encoder.layers.\(i).", w: w, spec: spec)
        }
        super.init()
    }

    /// mel [1, T, F] -> (features [1, Tout, D], lengths [1])
    func callAsFunction(_ x: MLXArray, lengths: MLXArray?) -> (MLXArray, MLXArray) {
        var outLengths = lengths ?? MLXArray.full([x.shape[0]], values: MLXArray(x.shape[1]), dtype: .int64)
        let (y, subsampledLengths) = self.preEncode(x, lengths: outLengths)

        let (scaled, posEmb) = self.posEnc(y, offset: 0)
        var h = scaled
        for layer in self.layers {
            h = layer(h, posEmb: posEmb)
        }
        return (h, subsampledLengths)
    }
}

// MARK: - Predict network (embed + LSTM x2)

nonisolated final class ParakeetMlxLSTM: Module {
    let hiddenSize: Int
    let wx: MLXArray // [4H, D]
    let wh: MLXArray // [4H, H]
    let bias: MLXArray?

    init(wx: MLXArray, wh: MLXArray, bias: MLXArray?) {
        self.wx = wx
        self.wh = wh
        self.bias = bias
        self.hiddenSize = wh.shape[1]
        super.init()
    }

    /// x: [B, T, D], h/c: [B*?] -> keep it simple: [1, H]
    func callAsFunction(
        _ x: MLXArray, hidden: MLXArray?, cell: MLXArray?
    ) -> (MLXArray, MLXArray, MLXArray) {
        var xp = matmul(x, self.wx.T) // [B, T, 4H]
        if let bias {
            xp = xp + bias
        }
        let t = xp.shape[1]
        var h = hidden ?? zeros([xp.shape[0], self.hiddenSize])
        var c = cell ?? zeros([xp.shape[0], self.hiddenSize])
        var allH: [MLXArray] = []
        var allC: [MLXArray] = []
        for idx in 0..<t {
            var ifgo = xp[0..., idx, 0...]
            ifgo = ifgo + matmul(h, self.wh.T)
            let parts = ifgo.reshaped([ifgo.shape[0], 4, self.hiddenSize])
            let i = sigmoid(parts[0..., 0, 0...])
            let f = sigmoid(parts[0..., 1, 0...])
            let g = tanh(parts[0..., 2, 0...])
            let o = sigmoid(parts[0..., 3, 0...])
            c = f * c + i * g
            h = o * tanh(c)
            allH.append(h)
            allC.append(c)
        }
        // Stack over time axis: python stacks at axis -2 producing [B, T, H]
        let hStack = stacked(allH, axis: 1)
        let cStack = stacked(allC, axis: 1)
        return (hStack, h, c)
    }
}

nonisolated final class ParakeetMlxPredictNetwork: Module {
    let predHidden: Int
    let embedWeight: MLXArray // [V+1, H]
    let lstm: [ParakeetMlxLSTM]

    /// Generic init (used by Nemotron with its own dims; same weight keys).
    init(embeddingCount: Int, predHidden: Int, predRNNLayers: Int, w: [String: MLXArray]) {
        self.predHidden = predHidden
        self.embedWeight = w["decoder.prediction.embed.weight"] ?? .zeros([embeddingCount, predHidden])
        var layers: [ParakeetMlxLSTM] = []
        for i in 0..<predRNNLayers {
            layers.append(ParakeetMlxLSTM(
                wx: w["decoder.prediction.dec_rnn.lstm.\(i).Wx"] ?? .zeros([4 * predHidden, predHidden]),
                wh: w["decoder.prediction.dec_rnn.lstm.\(i).Wh"] ?? .zeros([4 * predHidden, predHidden]),
                bias: w["decoder.prediction.dec_rnn.lstm.\(i).bias"]))
        }
        self.lstm = layers
        super.init()
    }

    init(spec: ParakeetMlxSpec, w: [String: MLXArray]) {
        self.predHidden = spec.predHidden
        self.embedWeight = w["decoder.prediction.embed.weight"] ?? .zeros([spec.decoderEmbeddingCount, spec.predHidden])
        var layers: [ParakeetMlxLSTM] = []
        for i in 0..<spec.predRNNLayers {
            layers.append(ParakeetMlxLSTM(
                wx: w["decoder.prediction.dec_rnn.lstm.\(i).Wx"] ?? .zeros([4 * spec.predHidden, spec.predHidden]),
                wh: w["decoder.prediction.dec_rnn.lstm.\(i).Wh"] ?? .zeros([4 * spec.predHidden, spec.predHidden]),
                bias: w["decoder.prediction.dec_rnn.lstm.\(i).bias"]))
        }
        self.lstm = layers
        super.init()
    }

    /// y: token id (Int) or nil for the start step.
    /// hidden/cell: per-layer arrays of [B, H] (layer count = predRNNLayers).
    /// Return (outputs [B, 1, H], hidden [B, H] per layer, cell per layer)
    func callAsFunction(
        y: Int?, hidden: [MLXArray?], cell: [MLXArray?]
    ) -> (MLXArray, [MLXArray], [MLXArray]) {
        var embedded: MLXArray
        if let y {
            let idx = MLXArray([Int32(y)]).reshaped([1, 1])
            embedded = take(self.embedWeight, idx, axis: 0)
        } else {
            embedded = zeros([1, 1, self.predHidden])
        }
        var newH: [MLXArray] = []
        var newC: [MLXArray] = []
        var layerInput = embedded
        for (i, layer) in self.lstm.enumerated() {
            let (outs, nh, nc) = layer(
                layerInput,
                hidden: i < hidden.count ? hidden[i] : nil,
                cell: i < cell.count ? cell[i] : nil)
            layerInput = outs
            newH.append(nh)
            newC.append(nc)
        }
        return (layerInput, newH, newC)
    }
}

// MARK: - Joint network

nonisolated final class ParakeetMlxJointNetwork: Module {
    let enc: ParakeetMlxLinear
    let pred: ParakeetMlxLinear
    let out: ParakeetMlxLinear
    let activation: (MLXArray) -> MLXArray

    /// Generic init (used by Nemotron with its own dims; same weight keys).
    init(encoderHidden: Int, predHidden: Int, jointHidden: Int, outputDim: Int, activation: String, w: [String: MLXArray]) {
        self.enc = ParakeetMlxLinear(
            weight: w["joint.enc.weight"] ?? .zeros([jointHidden, encoderHidden]),
            bias: w["joint.enc.bias"])
        self.pred = ParakeetMlxLinear(
            weight: w["joint.pred.weight"] ?? .zeros([jointHidden, predHidden]),
            bias: w["joint.pred.bias"])
        self.out = ParakeetMlxLinear(
            weight: w["joint.joint_net.2.weight"] ?? .zeros([outputDim, jointHidden]),
            bias: w["joint.joint_net.2.bias"])
        switch activation.lowercased() {
        case "sigmoid": self.activation = { sigmoid($0) }
        case "tanh": self.activation = { tanh($0) }
        default: self.activation = { maximum($0, 0) }
        }
        super.init()
    }

    init(spec: ParakeetMlxSpec, w: [String: MLXArray]) {
        self.enc = ParakeetMlxLinear(
            weight: w["joint.enc.weight"] ?? .zeros([spec.jointHidden, spec.encoderHidden]),
            bias: w["joint.enc.bias"])
        self.pred = ParakeetMlxLinear(
            weight: w["joint.pred.weight"] ?? .zeros([spec.jointHidden, spec.predHidden]),
            bias: w["joint.pred.bias"])
        self.out = ParakeetMlxLinear(
            weight: w["joint.joint_net.2.weight"] ?? .zeros([spec.jointOutputDim, spec.jointHidden]),
            bias: w["joint.joint_net.2.bias"])
        switch spec.jointActivation.lowercased() {
        case "sigmoid": self.activation = { sigmoid($0) }
        case "tanh": self.activation = { tanh($0) }
        default: self.activation = { maximum($0, 0) }
        }
        super.init()
    }

    /// enc: [B, T, D_enc], pred: [B, 1, D_pred] -> [B, T, 1, V+1+extra]
    func callAsFunction(_ enc: MLXArray, _ pred: MLXArray) -> MLXArray {
        let e = self.enc(enc).expandedDimensions(axis: 2) // [B, T, 1, J]
        let p = self.pred(pred).expandedDimensions(axis: 1) // [B, 1, 1, J]
        return self.out(self.activation(e + p))
    }
}

// MARK: - Full TDT model

nonisolated final class ParakeetMlxModel: Module {
    let spec: ParakeetMlxSpec
    let encoder: ParakeetMlxConformer
    let decoder: ParakeetMlxPredictNetwork
    let joint: ParakeetMlxJointNetwork

    init(spec: ParakeetMlxSpec, w: [String: MLXArray]) {
        self.spec = spec
        self.encoder = ParakeetMlxConformer(spec: spec, w: w)
        self.decoder = ParakeetMlxPredictNetwork(spec: spec, w: w)
        self.joint = ParakeetMlxJointNetwork(spec: spec, w: w)
        super.init()
    }

    func pnl(_ logits: MLXArray) -> [Float] {
        (0..<logits.shape[0]).map { logits[$0].item(Float.self) }
    }
}
