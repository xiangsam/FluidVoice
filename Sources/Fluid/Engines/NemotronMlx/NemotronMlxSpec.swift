import Foundation

// MARK: - Nemotron 3.5 ASR (streaming 0.6B) MLX spec
// Mirrors mlx_audio NemotronASRConfig (NeMo EncDecRNNTBPEModelWithPrompt).

struct NemotronMlxSpec {
    // Preprocessor (normalize: "NA" -> no normalization)
    let sampleRate: Int
    let features: Int
    let nFFT: Int
    let windowSize: Double
    let windowStride: Double
    let preemph: Float
    let logGuard: Float

    // Encoder (cache-aware FastConformer)
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

    // Prompt
    let numPrompts: Int
    let promptHidden: Int
    let promptDictionary: [String: Int]
    let defaultLanguage: String

    // Decoder / joint
    let decoderVocabSize: Int
    let blankAsPad: Bool
    let predHidden: Int
    let predRNNLayers: Int
    let jointHidden: Int
    let jointActivation: String
    let encoderHidden: Int

    // Decoding
    let maxSymbols: Int
    let attContextSize: [Int] // [left, right], default [56, 13]

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
    /// Output freq dim after causal 3x3 stride-2 convs (pad_left=2, pad_right=1).
    var finalFreqDim: Int {
        var f = self.featIn
        for _ in 0..<self.subsamplingLayers {
            f = (f + 2 + 1 - 3) / 2 + 1
        }
        return f
    }
    var subsamplingOutputDim: Int {
        self.subsamplingConvChannels * self.finalFreqDim
    }
    var jointOutputDim: Int { self.decoderVocabSize + 1 }
    var blankID: Int { self.decoderVocabSize }
    var leftContext: Int { self.attContextSize.first ?? 56 }
    var rightContext: Int { self.attContextSize.count > 1 ? self.attContextSize[1] : 13 }
    /// Native encoder chunk size = right_context + 1 (encoder frames).
    var chunkFrames: Int { self.rightContext + 1 }
    /// Mel frames per encoder chunk.
    var chunkMelFrames: Int { self.chunkFrames * self.subsamplingFactor }

    func promptIndex(language: String?) -> Int {
        let lang = language ?? self.defaultLanguage
        if let idx = self.promptDictionary[lang] { return idx }
        if let idx = self.promptDictionary[self.defaultLanguage] { return idx }
        return 0
    }

    // MARK: - decode helpers (SentencePiece pieces in config.json)

    static let otherSpecial: Set<String> = ["<unk>", "<pad>", "<s>", "</s>"]

    func isLangTag(_ piece: String) -> Bool {
        piece.count >= 4 && piece.hasPrefix("<") && piece.hasSuffix(">")
            && piece.dropFirst().dropLast().contains("-")
    }

    func isSpecialPiece(_ piece: String) -> Bool {
        Self.otherSpecial.contains(piece) || self.isLangTag(piece)
    }

    /// Decode token ids, skipping special tokens and language tags.
    func decode(tokens: [Int], stripLangTags: Bool = true) -> String {
        var parts: [String] = []
        for token in tokens {
            if token < 0 || token >= self.vocabulary.count { continue }
            let piece = self.vocabulary[token]
            if Self.otherSpecial.contains(piece) { continue }
            if stripLangTags && self.isLangTag(piece) { continue }
            parts.append(piece.replacingOccurrences(of: "▁", with: " "))
        }
        return parts.joined()
    }
}

enum NemotronMlxError: LocalizedError {
    case invalidWeights(String)
    case notLoaded
    case invalidAudio

    var errorDescription: String? {
        switch self {
        case .invalidWeights(let m): return "Nemotron MLX weights error: \(m)"
        case .notLoaded: return "Nemotron MLX model not loaded"
        case .invalidAudio: return "Invalid audio"
        }
    }
}
