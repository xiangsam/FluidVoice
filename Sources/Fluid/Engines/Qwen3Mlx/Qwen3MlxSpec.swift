import Foundation

// MARK: - Qwen3-ASR MLX Engine Configuration
//
// Architecture values are read from the downloaded config.json so the same
// engine serves every mlx-community variant (0.6B / 1.7B, bf16 / quantized).

struct Qwen3MlxSpec {
    // Audio encoder (Whisper-style; "audio_config" from config.json)
    let encDModel: Int
    let encLayers: Int
    let encHeads: Int
    let encFFNDim: Int
    let numMelBins: Int
    let maxSourcePositions: Int
    let encOutputDim: Int
    let nWindow: Int
    let nWindowInfer: Int
    let convChunkSize: Int
    let downsampleHidden: Int

    // Text decoder ("text_config" from config.json)
    let hiddenSize: Int
    let numLayers: Int
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let intermediateSize: Int
    let vocabSize: Int
    let rmsEps: Float
    let ropeTheta: Float

    var chunkSize: Int { nWindow * 2 }

    // Qwen3-ASR-0.6B (matches mlx-community/Qwen3-ASR-0.6B-* config.json)
    static let qwen3Asr0p6B = Qwen3MlxSpec(
        encDModel: 896, encLayers: 18, encHeads: 14, encFFNDim: 3584,
        numMelBins: 128, maxSourcePositions: 1500, encOutputDim: 1024,
        nWindow: 50, nWindowInfer: 800, convChunkSize: 500, downsampleHidden: 480,
        hiddenSize: 1024, numLayers: 28, numHeads: 16, numKVHeads: 8, headDim: 128,
        intermediateSize: 3072, vocabSize: 151_936, rmsEps: 1e-6, ropeTheta: 1_000_000
    )

    static func parse(from config: [String: Any]) throws -> Qwen3MlxSpec {
        let thinker = (config["thinker_config"] as? [String: Any]) ?? config
        let audio = (thinker["audio_config"] as? [String: Any]) ?? config
        let text = (thinker["text_config"] as? [String: Any])
            ?? (config["text_decoder_config"] as? [String: Any])
            ?? config

        func int(_ dict: [String: Any], _ keys: [String], default def: Int) -> Int {
            for k in keys {
                if let v = dict[k] as? Int { return v }
                if let v = dict[k] as? Double { return Int(v) }
            }
            return def
        }
        func float(_ dict: [String: Any], _ keys: [String], default def: Float) -> Float {
            for k in keys {
                if let v = dict[k] as? Double { return Float(v) }
                if let v = dict[k] as? NSNumber { return v.floatValue }
            }
            return def
        }

        let vocab = int(text, ["vocab_size"], default: 0)
        guard vocab > 0 else {
            throw Qwen3MlxError.invalidWeights("config.json missing vocab_size")
        }

        return Qwen3MlxSpec(
            encDModel: int(audio, ["d_model"], default: 896),
            encLayers: int(audio, ["encoder_layers", "num_hidden_layers"], default: 18),
            encHeads: int(audio, ["encoder_attention_heads"], default: 14),
            encFFNDim: int(audio, ["encoder_ffn_dim"], default: 3584),
            numMelBins: int(audio, ["num_mel_bins"], default: 128),
            maxSourcePositions: int(audio, ["max_source_positions"], default: 1500),
            encOutputDim: int(audio, ["output_dim"], default: 1024),
            nWindow: int(audio, ["n_window"], default: 50),
            nWindowInfer: int(audio, ["n_window_infer"], default: 800),
            convChunkSize: int(audio, ["conv_chunksize"], default: 500),
            downsampleHidden: int(audio, ["downsample_hidden_size"], default: 480),
            hiddenSize: int(text, ["hidden_size"], default: 1024),
            numLayers: int(text, ["num_hidden_layers"], default: 28),
            numHeads: int(text, ["num_attention_heads"], default: 16),
            numKVHeads: int(text, ["num_key_value_heads"], default: 8),
            headDim: int(text, ["head_dim"], default: 128),
            intermediateSize: int(text, ["intermediate_size"], default: 3072),
            vocabSize: vocab,
            rmsEps: float(text, ["rms_norm_eps"], default: 1e-6),
            ropeTheta: float(text, ["rope_theta"], default: 1_000_000)
        )
    }
}

// MARK: - Special tokens (identical across Qwen3-ASR variants)

enum Qwen3MlxTokens {
    static let audioStartTokenId = 151_669
    static let audioEndTokenId = 151_670
    static let audioPadTokenId = 151_676
    static let asrTextTokenId = 151_704
    static let imStartTokenId = 151_644
    static let imEndTokenId = 151_645
    static let endOfTextTokenId = 151_643
    static let systemTokenId = 8948
    static let userTokenId = 872
    static let assistantTokenId = 77_091
    static let newlineTokenId = 198
    static let languageTokenId = 11528

    static let eosTokenIds: Set<Int> = [endOfTextTokenId, imEndTokenId]
}

// MARK: - Quantization info (from config.json)

struct Qwen3MlxQuantInfo {
    let groupSize: Int
    let bits: Int

    static func parse(from config: [String: Any]) -> Qwen3MlxQuantInfo? {
        let q = (config["quantization"] as? [String: Any])
            ?? (config["quantization_config"] as? [String: Any])
        guard let q else { return nil }
        let bits = (q["bits"] as? Int) ?? (q["bits"] as? NSNumber)?.intValue
        let groupSize = (q["group_size"] as? Int) ?? (q["group_size"] as? NSNumber)?.intValue
        guard let bits, let groupSize else { return nil }
        return Qwen3MlxQuantInfo(groupSize: groupSize, bits: bits)
    }
}
