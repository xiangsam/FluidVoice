import Foundation

// MARK: - GLM-ASR-Nano MLX spec (Whisper encoder + LLaMA decoder + MLP adapter)

struct GlmMlxSpec {
    // Audio
    let sampleRate: Int
    let numMelBins: Int

    // Whisper encoder
    let encDModel: Int
    let encHeads: Int
    let encFFNDim: Int
    let encLayers: Int
    let maxSourcePositions: Int
    let useRope: Bool
    let ropeTraditional: Bool

    // Adapter
    let mergeFactor: Int
    let adapterInitialized: Bool
    let adapterIntermediate: Int

    // LLaMA decoder
    let vocabSize: Int
    let hiddenSize: Int
    let intermediateSize: Int
    let numLayers: Int
    let numHeads: Int
    let numKVHeads: Int
    let rmsEps: Float
    let ropeTheta: Float
    let headDim: Int

    let eosTokenIds: [Int]

    var encHeadDim: Int { self.encDModel / self.encHeads }
    var ropeDim: Int { self.encHeadDim / 2 } // whisper halves the rotary dim

    static func parse(from config: [String: Any]) throws -> GlmMlxSpec {
        func int(_ d: [String: Any]?, _ k: String, _ def: Int) -> Int {
            (d?[k] as? NSNumber)?.intValue ?? def
        }
        func float(_ d: [String: Any]?, _ k: String, _ def: Float) -> Float {
            (d?[k] as? NSNumber)?.floatValue ?? def
        }
        func bool(_ d: [String: Any]?, _ k: String, _ def: Bool) -> Bool {
            (d?[k] as? NSNumber)?.boolValue ?? def
        }

        let w = config["whisper_config"] as? [String: Any]
        let lm = config["lm_config"] as? [String: Any]
        let eosRaw = (lm?["eos_token_id"] as? [Int]) ?? [59246, 59253, 59255]

        return GlmMlxSpec(
            sampleRate: int(config, "sample_rate", 16_000),
            numMelBins: int(w, "num_mel_bins", 128),
            encDModel: int(w, "d_model", 1280),
            encHeads: int(w, "encoder_attention_heads", 20),
            encFFNDim: int(w, "encoder_ffn_dim", 5120),
            encLayers: int(w, "encoder_layers", 32),
            maxSourcePositions: int(w, "max_source_positions", 1500),
            useRope: bool(config, "use_rope", true),
            ropeTraditional: bool(w, "rope_traditional", true),
            mergeFactor: int(config, "merge_factor", 4),
            adapterInitialized: (config["adapter_type"] as? String) != nil || config["merge_factor"] != nil,
            adapterIntermediate: int(config, "mlp_adapter_act", 0), // unused flag slot
            vocabSize: int(lm, "vocab_size", 59264),
            hiddenSize: int(lm, "hidden_size", 2048),
            intermediateSize: int(lm, "intermediate_size", 6144),
            numLayers: int(lm, "num_hidden_layers", 28),
            numHeads: int(lm, "num_attention_heads", 16),
            numKVHeads: int(lm, "num_key_value_heads", 4),
            rmsEps: float(lm, "rms_norm_eps", 1e-5),
            ropeTheta: float(lm, "rope_theta", 10_000),
            headDim: int(lm, "head_dim", 128),
            eosTokenIds: eosRaw
        )
    }
}

enum GlmMlxError: LocalizedError {
    case invalidWeights(String)
    case notLoaded
    case invalidAudio

    var errorDescription: String? {
        switch self {
        case .invalidWeights(let m): return "GLM-ASR MLX weights error: \(m)"
        case .notLoaded: return "GLM-ASR MLX model not loaded"
        case .invalidAudio: return "Invalid audio"
        }
    }
}
