import Foundation

// MARK: - Fun-ASR (MLT) Nano MLX spec (frontend + SANM encoder + adaptor + Qwen3 LLM)

struct FunMlxSpec {
    // Frontend (Kaldi fbank + LFR)
    let fs: Int
    let winType: String
    let nMel: Int
    let frameLength: Int // ms
    let frameShift: Int // ms
    let lfrM: Int
    let lfrN: Int
    let preemph: Float
    let lowFreq: Float

    // SANM audio encoder
    let encOutputSize: Int
    let encHeads: Int
    let encLinearUnits: Int
    let encBlocks: Int
    let encTPBlocks: Int
    let encKernel: Int
    let encSanmShift: Int

    // Adaptor
    let adaptorFFNDim: Int
    let adaptorLLMDim: Int
    let adaptorEncoderDim: Int
    let adaptorLayers: Int
    let adaptorHeads: Int

    // LLM (Qwen3-0.6B)
    let vocabSize: Int
    let hiddenSize: Int
    let intermediateSize: Int
    let numLayers: Int
    let numHeads: Int
    let numKVHeads: Int
    let rmsEps: Float
    let ropeTheta: Float
    let headDim: Int

    var winLen: Int { self.fs * self.frameLength / 1000 }
    var winInc: Int { self.fs * self.frameShift / 1000 }
    var encDModel: Int { self.encOutputSize }

    static func parse(from config: [String: Any]) throws -> FunMlxSpec {
        func int(_ d: [String: Any]?, _ k: String, _ def: Int) -> Int {
            (d?[k] as? NSNumber)?.intValue ?? def
        }
        func float(_ d: [String: Any]?, _ k: String, _ def: Float) -> Float {
            (d?[k] as? NSNumber)?.floatValue ?? def
        }
        func str(_ d: [String: Any]?, _ k: String, _ def: String) -> String {
            (d?[k] as? String) ?? def
        }

        // mlx-community Fun-ASR (converted with mlx-audio-plus) ships a flat
        // config: top-level sample_rate/n_mels/lfr_m/lfr_n + encoder/adaptor/
        // llm dicts. Older configs (fun_asr_nano) used nested frontend_conf/
        // audio_encoder_conf/audio_adaptor_conf/text_config; keep as fallback.
        let fe = (config["frontend_conf"] as? [String: Any]) ?? config
        let enc = (config["audio_encoder_conf"] as? [String: Any])
            ?? (config["encoder"] as? [String: Any])
        let ad = (config["audio_adaptor_conf"] as? [String: Any])
            ?? (config["adaptor"] as? [String: Any])
        let lm = (config["text_config"] as? [String: Any])
            ?? (config["llm"] as? [String: Any])

        let hidden = int(lm, "hidden_size", 1024)
        let heads = int(lm, "num_attention_heads", 16)
        let headDim = int(lm, "head_dim", 128)
        let enc0 = int(enc, "num_encoders0", 1)
        let encN = int(enc, "num_encoders", 49)

        return FunMlxSpec(
            fs: int(fe, "sample_rate", int(fe, "fs", 16_000)),
            winType: str(fe, "window", "hamming"),
            nMel: int(fe, "n_mels", 80),
            frameLength: int(fe, "frame_length", 25),
            frameShift: int(fe, "frame_shift", 10),
            lfrM: int(fe, "lfr_m", 7),
            lfrN: int(fe, "lfr_n", 6),
            preemph: float(fe, "preemphasis", 0.97),
            lowFreq: float(fe, "low_freq", 20.0),
            encOutputSize: int(enc, "output_size", int(enc, "encoder_dim", 512)),
            encHeads: int(enc, "attention_heads", int(enc, "num_heads", 4)),
            encLinearUnits: int(enc, "linear_units", int(enc, "ffn_dim", 2048)),
            encBlocks: int(enc, "num_blocks", enc0 + encN),
            encTPBlocks: int(enc, "tp_blocks", int(enc, "num_tp_encoders", 20)),
            encKernel: int(enc, "kernel_size", 11),
            encSanmShift: int(enc, "sanm_shift", 0),
            adaptorFFNDim: int(ad, "ffn_dim", 2048),
            adaptorLLMDim: int(ad, "llm_dim", 1024),
            adaptorEncoderDim: int(ad, "encoder_dim", 512),
            adaptorLayers: int(ad, "n_layer", 2),
            adaptorHeads: int(ad, "attention_heads", 8),
            vocabSize: int(lm, "vocab_size", 151936),
            hiddenSize: hidden,
            intermediateSize: int(lm, "intermediate_size", 4096),
            numLayers: int(lm, "num_hidden_layers", 28),
            numHeads: heads,
            numKVHeads: int(lm, "num_key_value_heads", 8),
            rmsEps: float(lm, "rms_norm_eps", 1e-6),
            ropeTheta: float(lm, "rope_theta", 1_000_000),
            headDim: headDim
        )
    }
}

enum FunMlxError: LocalizedError {
    case invalidWeights(String)
    case notLoaded
    case invalidAudio

    var errorDescription: String? {
        switch self {
        case .invalidWeights(let m): return "Fun-ASR MLX weights error: \(m)"
        case .notLoaded: return "Fun-ASR MLX model not loaded"
        case .invalidAudio: return "Invalid audio"
        }
    }
}
