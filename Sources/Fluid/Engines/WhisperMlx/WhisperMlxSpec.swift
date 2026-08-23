import Foundation

// MARK: - Whisper MLX spec (mlx-audio whisper port)
//
// Config is the flat whisper config (n_mels / n_audio_* / n_text_* / n_vocab),
// plus an optional quantization block (bits + group_size).

struct WhisperMlxSpec {
    let nMels: Int
    let audioCtx: Int
    let audioState: Int
    let audioHeads: Int
    let audioLayers: Int
    let vocabSize: Int
    let textCtx: Int
    let textState: Int
    let textHeads: Int
    let textLayers: Int
    let quantBits: Int
    let quantGroupSize: Int

    static func parse(from config: [String: Any]) -> WhisperMlxSpec {
        func int(_ d: [String: Any]?, _ k: String, _ def: Int) -> Int {
            (d?[k] as? NSNumber)?.intValue ?? def
        }
        // MLX-format configs use n_mels / n_audio_state / ...; HuggingFace
        // configs use num_mel_bins / d_model / encoder_layers / ... .
        let q = (config["quantization"] as? [String: Any])
            ?? (config["quantization_config"] as? [String: Any])
        let dModel = int(config, "d_model", int(config, "n_audio_state", 1280))
        return WhisperMlxSpec(
            nMels: int(config, "n_mels", int(config, "num_mel_bins", 80)),
            audioCtx: int(config, "n_audio_ctx", int(config, "max_source_positions", 1500)),
            audioState: dModel,
            audioHeads: int(config, "n_audio_head", int(config, "encoder_attention_heads", 20)),
            audioLayers: int(config, "n_audio_layer", int(config, "encoder_layers", 32)),
            vocabSize: int(config, "n_vocab", int(config, "vocab_size", 51866)),
            textCtx: int(config, "n_text_ctx", int(config, "max_target_positions", 448)),
            textState: int(config, "n_text_state", dModel),
            textHeads: int(config, "n_text_head", int(config, "decoder_attention_heads", 20)),
            textLayers: int(config, "n_text_layer", int(config, "decoder_layers", 4)),
            quantBits: int(q, "bits", 8),
            quantGroupSize: int(q, "group_size", 64)
        )
    }
}

enum WhisperMlxError: LocalizedError {
    case invalidWeights(String)
    case notLoaded
    case invalidAudio

    var errorDescription: String? {
        switch self {
        case .invalidWeights(let m): return "Whisper MLX weights error: \(m)"
        case .notLoaded: return "Whisper MLX model not loaded"
        case .invalidAudio: return "Invalid audio"
        }
    }
}

/// Whisper special token ids (multilingual vocab >= 51865).
enum WhisperTokens {
    static let eot = 50257
    static let sot = 50258
    static var langStart: Int { 50259 }
    static let transcribe = 50359
    static let translate = 50358
    static let noTimestamps = 50363
    static let blank = 220

    /// Canonical language order matches the whisper multilingual tokenizer.
    static let languages: [String] = [
        "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca",
        "nl", "ar", "sv", "it", "id", "hi", "fi", "vi", "he", "uk", "el", "ms",
        "cs", "ro", "da", "hu", "ta", "no", "th", "ur", "hr", "bg", "lt", "la",
        "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn",
        "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw",
        "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc", "ka", "be",
        "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn",
        "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln", "ha",
        "ba", "jw", "su", "yue",
    ]

    static func languageToken(_ code: String) -> Int {
        let idx = languages.firstIndex(of: code.lowercased()) ?? 0
        return langStart + idx
    }
}
