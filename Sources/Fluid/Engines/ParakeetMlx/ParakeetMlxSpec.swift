import Foundation

// MARK: - Parakeet TDT (NVIDIA) MLX spec
//
// Parses the NeMo-style config.json of mlx-community/parakeet-tdt-0.6b-v3
// (and compatible TDT models) into the parameters the Swift engine needs.

struct ParakeetMlxSpec {
    // Preprocessor (audio.py::PreprocessArgs)
    let sampleRate: Int
    let features: Int // mel bins
    let nFFT: Int
    let windowSize: Double
    let windowStride: Double
    let preemph: Float
    let magPower: Float

    // Encoder (conformer)
    let featIn: Int
    let dModel: Int
    let nLayers: Int
    let nHeads: Int
    let ffExpansionFactor: Int
    let subsamplingFactor: Int
    let subsamplingConvChannels: Int
    let convKernelSize: Int
    let posEmbMaxLen: Int
    let useBias: Bool
    let xscaling: Bool

    // Decoder (predict network)
    let decoderVocabSize: Int
    let blankAsPad: Bool
    let predHidden: Int
    let predRNNLayers: Int

    // Joint
    let jointHidden: Int
    let jointActivation: String
    let encoderHidden: Int
    let numExtraOutputs: Int

    // Decoding (TDT)
    let durations: [Int]
    let maxSymbols: Int

    // Vocabulary (8192 tokens, embedded in config.json)
    let vocabulary: [String]

    var winLength: Int { Int(self.windowSize * Double(self.sampleRate)) }
    var hopLength: Int { Int(self.windowStride * Double(self.sampleRate)) }
    var headDim: Int { self.dModel / self.nHeads }
    var subsamplingLayers: Int {
        var f = 0
        var v = self.subsamplingFactor
        while v > 1 { f += 1; v >>= 1 }
        return f
    }
    /// Frequency dim after subsampling convs (channels-last conv on [T, F, 1]).
    var finalFreqDim: Int {
        var f = self.featIn
        let pad = 1
        for _ in 0..<self.subsamplingLayers {
            f = (f + 2 * pad - 3) / 2 + 1
        }
        return f
    }
    /// Token output dim after subsampling: channels * finalFreqDim.
    var subsamplingOutputDim: Int {
        self.subsamplingConvChannels * self.finalFreqDim
    }
    var jointOutputDim: Int {
        self.decoderVocabSize + 1 + self.numExtraOutputs
    }
    var decoderEmbeddingCount: Int {
        self.blankAsPad ? self.decoderVocabSize + 1 : self.decoderVocabSize
    }

    static func parse(from config: [String: Any]) throws -> ParakeetMlxSpec {
        let pre = config["preprocessor"] as? [String: Any]
        let enc = config["encoder"] as? [String: Any]
        let dec = config["decoder"] as? [String: Any]
        let pred = dec?["prednet"] as? [String: Any]
        let joint = config["joint"] as? [String: Any]
        let jn = joint?["jointnet"] as? [String: Any]
        let decoding = config["decoding"] as? [String: Any]
        let greedy = decoding?["greedy"] as? [String: Any]

        func int(_ d: [String: Any]?, _ k: String, _ def: Int) -> Int {
            (d?[k] as? NSNumber)?.intValue ?? def
        }
        func float(_ d: [String: Any]?, _ k: String, _ def: Float) -> Float {
            (d?[k] as? NSNumber)?.floatValue ?? def
        }
        func double(_ d: [String: Any]?, _ k: String, _ def: Double) -> Double {
            (d?[k] as? NSNumber)?.doubleValue ?? def
        }
        func bool(_ d: [String: Any]?, _ k: String, _ def: Bool) -> Bool {
            (d?[k] as? NSNumber)?.boolValue ?? def
        }

        guard let vocab = joint?["vocabulary"] as? [String] else {
            throw ParakeetMlxError.invalidWeights("Missing joint.vocabulary")
        }

        // dacite falls back to dataclass defaults for missing keys:
        // PreprocessArgs.preemph = 0.97, mag_power = 2.0
        let spec = ParakeetMlxSpec(
            sampleRate: int(pre, "sample_rate", 16_000),
            features: int(pre, "features", 128),
            nFFT: int(pre, "n_fft", 512),
            windowSize: double(pre, "window_size", 0.025),
            windowStride: double(pre, "window_stride", 0.01),
            preemph: (pre?["preemph"] as? NSNumber)?.floatValue ?? 0.97,
            magPower: (pre?["mag_power"] as? NSNumber)?.floatValue ?? 2.0,
            featIn: int(enc, "feat_in", 128),
            dModel: int(enc, "d_model", 1024),
            nLayers: int(enc, "n_layers", 24),
            nHeads: int(enc, "n_heads", 8),
            ffExpansionFactor: int(enc, "ff_expansion_factor", 4),
            subsamplingFactor: int(enc, "subsampling_factor", 8),
            subsamplingConvChannels: int(enc, "subsampling_conv_channels", 256),
            convKernelSize: int(enc, "conv_kernel_size", 9),
            posEmbMaxLen: int(enc, "pos_emb_max_len", 5000),
            useBias: bool(enc, "use_bias", false),
            xscaling: bool(enc, "xscaling", false),
            decoderVocabSize: int(dec, "vocab_size", 8192),
            blankAsPad: bool(dec, "blank_as_pad", true),
            predHidden: int(pred, "pred_hidden", 640),
            predRNNLayers: int(pred, "pred_rnn_layers", 2),
            jointHidden: int(jn, "joint_hidden", 640),
            jointActivation: (jn?["activation"] as? String) ?? "relu",
            encoderHidden: int(jn, "encoder_hidden", 1024),
            numExtraOutputs: int(joint, "num_extra_outputs", 5),
            durations: ((config["decoding"] as? [String: Any])?["durations"] as? [Int]) ?? [0, 1, 2, 3, 4],
            maxSymbols: int(greedy, "max_symbols", 10),
            vocabulary: vocab
        )
        return spec
    }

    // MARK: - Decode helpers

    /// Join token ids into text (parakeet-mlx tokenizer.decode semantics).
    func decode(tokens: [Int]) -> String {
        tokens.map { self.vocabulary[$0].replacingOccurrences(of: "▁", with: " ") }
            .joined()
    }
}

enum ParakeetMlxError: LocalizedError {
    case invalidWeights(String)
    case notLoaded
    case invalidAudio

    var errorDescription: String? {
        switch self {
        case .invalidWeights(let m): return "Parakeet MLX weights error: \(m)"
        case .notLoaded: return "Parakeet MLX model not loaded"
        case .invalidAudio: return "Invalid audio"
        }
    }
}
