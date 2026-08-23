import Foundation
import MLX

// MARK: - Whisper MLX engine (mlx-audio whisper port)

actor WhisperMlxEngine {
    private var model: WhisperMlxModel?
    private var spec: WhisperMlxSpec?
    private var tokenizer: WhisperMlxTokenizer?
    private var mel: WhisperMlxMel?
    private var currentDirectory: URL?

    static func modelsRootDirectory() -> URL { Qwen3MlxEngine.modelsRootDirectory() }

    static func cacheDirectory(modelID: String, variantID: String) -> URL {
        modelsRootDirectory()
            .appendingPathComponent("MlxStt", isDirectory: true)
            .appendingPathComponent(modelID, isDirectory: true)
            .appendingPathComponent(variantID, isDirectory: true)
    }

    static func modelsExist(modelID: String, variantID: String) -> Bool {
        guard MlxSttCatalog.card(pathID: "\(modelID)/\(variantID)") != nil else { return false }
        let dir = cacheDirectory(modelID: modelID, variantID: variantID)
        let weights = dir.appendingPathComponent("model.safetensors")
        guard FileManager.default.fileExists(atPath: weights.path) else { return false }
        guard let size = try? FileManager.default.attributesOfItem(atPath: weights.path)[.size] as? Int,
            size > 100_000_000
        else { return false }
        // Reject half-written weight files left by interrupted downloads.
        guard MlxModelDownloader.isValidSafetensors(at: weights) else { return false }
        return FileManager.default.fileExists(
            atPath: dir.appendingPathComponent("config.json").path)
    }

    // MARK: - Download

    static func download(
        modelID: String, variantID: String,
        progressHandler: ((Double) -> Void)? = nil
    ) async throws -> URL {
        guard let card = MlxSttCatalog.card(pathID: "\(modelID)/\(variantID)") else {
            throw WhisperMlxError.invalidWeights("Unknown Whisper card \(modelID)/\(variantID)")
        }
        let directory = cacheDirectory(modelID: modelID, variantID: variantID)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let fm = FileManager.default
        let base = SettingsStore.shared.huggingFaceBaseURL

        func fetchData(_ relativePath: String) async throws -> Data {
            guard let url = URL(string: "\(base)/\(card.repo)/resolve/main/\(relativePath)") else {
                throw URLError(.badURL)
            }
            var request = URLRequest(url: url)
            request.timeoutInterval = 60
            let (data, _) = try await URLSession.shared.data(for: request)
            guard !data.isEmpty else { throw URLError(.zeroByteResource) }
            return data
        }

        // Files to fetch from the card's own repo. MLX whisper repos are
        // weight-only (model.safetensors + config.json); the tokenizer usually
        // lives in the matching openai/whisper-<size> HF repo.
        let sideFiles: [String]
        if card.repo.contains("turbo") {
            sideFiles = ["config.json", "multilingual.tiktoken"]
        } else {
            sideFiles = ["config.json"]
        }
        for file in sideFiles {
            let dest = directory.appendingPathComponent(file)
            if !fm.fileExists(atPath: dest.path) {
                let data = try await fetchData(file)
                try data.write(to: dest)
            }
        }

        // Tokenizer: prefer openai/whisper-<size> (all sizes have
        // multilingual.tiktoken there); fall back to tokenizer.json when the
        // card repo ships one.
        let tiktokenURL = directory.appendingPathComponent("multilingual.tiktoken")
        let tokenizerJsonURL = directory.appendingPathComponent("tokenizer.json")
        let vocabURL = directory.appendingPathComponent("vocab.json")
        let mergesURL = directory.appendingPathComponent("merges.txt")
        if !fm.fileExists(atPath: tiktokenURL.path), !fm.fileExists(atPath: tokenizerJsonURL.path) {
            // Derive the openai repo name from the card's config / spec.
            let openaiRepo = Self.openaiWhisperRepoName(for: card)
            var tokenizerFetched = false
            if let repo = openaiRepo {
                // The openai/whisper-* HF repos ship tokenizer.json (and
                // vocab.json/merges.txt) but NOT multilingual.tiktoken — that
                // file only exists in the mlx-community turbo repos. Try
                // tokenizer.json first, then the tiktoken variant as a fallback.
                for candidate in ["tokenizer.json", "multilingual.tiktoken"] {
                    guard let url = URL(string: "\(base)/\(repo)/resolve/main/\(candidate)") else { continue }
                    if let (data, _) = try? await URLSession.shared.data(from: url), !data.isEmpty {
                        // Guard against HF "Entry not found" text bodies.
                        let preview = String(data: data.prefix(16), encoding: .utf8) ?? ""
                        if preview == "Entry not found" { continue }
                        try data.write(to: candidate == "multilingual.tiktoken" ? tiktokenURL : tokenizerJsonURL)
                        tokenizerFetched = true
                        break
                    }
                }
            }
            if !tokenizerFetched {
                throw WhisperMlxError.invalidWeights(
                    "No tokenizer available for \(card.repo) (tried openai/whisper-<size> raw files)"
                )
            }
        }

        let weightsDest = directory.appendingPathComponent("model.safetensors")
        if !fm.fileExists(atPath: weightsDest.path)
            || ((try? (fm.attributesOfItem(atPath: weightsDest.path)[.size] as? Int) ?? 0) ?? 0) < 100_000_000
        {
            guard let url = URL(string: "\(base)/\(card.repo)/resolve/main/model.safetensors") else {
                throw URLError(.badURL)
            }
            try await MlxModelDownloader.downloadFile(
                from: url,
                to: weightsDest,
                progress: { fraction in
                    Task { @MainActor in
                        progressHandler?(fraction)
                    }
                }
            )
        }
        progressHandler?(1.0)
        return directory
    }

    /// Map a whisper card to the matching openai/whisper-<size> HF repo that
    /// ships the multilingual tokenizer. Used when the MLX weight repo does not
    /// include tokenizer files.
    static func openaiWhisperRepoName(for card: MlxSttCard) -> String? {
        switch card.specID {
        case "Tiny": return "openai/whisper-tiny"
        case "Base": return "openai/whisper-base"
        case "Small": return "openai/whisper-small"
        case "Medium": return "openai/whisper-medium"
        case "Large V3": return "openai/whisper-large-v3"
        case "Large V3 Turbo": return "openai/whisper-large-v3-turbo"
        case "Large V2": return "openai/whisper-large-v2"
        default: return nil
        }
    }

    // MARK: - Load

    func loadModels(modelID: String, variantID: String, directory: URL? = nil) async throws {
        let dir = directory ?? Self.cacheDirectory(modelID: modelID, variantID: variantID)
        let configData = try Data(contentsOf: dir.appendingPathComponent("config.json"))
        guard let config = try JSONSerialization.jsonObject(with: configData) as? [String: Any] else {
            throw WhisperMlxError.invalidWeights("Bad config.json")
        }
        let spec = WhisperMlxSpec.parse(from: config)

        let weightsURL = dir.appendingPathComponent("model.safetensors")
        let arrays = try MLX.loadArrays(url: weightsURL)
        let raw: [String: MLXArray] = Dictionary(uniqueKeysWithValues: arrays.map { ($0, $1) })
        let w = Self.remapWeights(raw)

        let tokenizer: WhisperMlxTokenizer
        let tiktokenURL = dir.appendingPathComponent("multilingual.tiktoken")
        if FileManager.default.fileExists(atPath: tiktokenURL.path) {
            tokenizer = try WhisperMlxTokenizer(tiktokenFile: tiktokenURL)
        } else {
            tokenizer = try WhisperMlxTokenizer(tokenizerJson: dir.appendingPathComponent("tokenizer.json"))
        }

        self.model = WhisperMlxModel(spec: spec, w: w)
        self.spec = spec
        self.tokenizer = tokenizer
        self.mel = WhisperMlxMel(spec: spec)
        self.currentDirectory = dir
    }

    /// Map HF-format keys (model.encoder... / decoder.embed_tokens...) to the
    /// MLX layout used by WhisperMlxModel; MLX-format repos pass through.
    static func remapWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        let isHF = weights.keys.contains { $0.hasPrefix("model.") }
        guard isHF else { return weights }

        let keyMap: [(String, String?)] = [
            ("encoder.embed_positions.weight", nil),
            ("decoder.embed_positions.weight", "decoder.positional_embedding"),
            ("encoder.layer_norm.", "encoder.ln_post."),
            ("decoder.layer_norm.", "decoder.ln."),
            ("encoder.layers.", "encoder.blocks."),
            ("decoder.layers.", "decoder.blocks."),
            (".self_attn_layer_norm.", ".attn_ln."),
            (".final_layer_norm.", ".mlp_ln."),
            (".encoder_attn_layer_norm.", ".cross_attn_ln."),
            (".fc1.", ".mlp1."),
            (".fc2.", ".mlp2."),
            (".self_attn.q_proj.", ".attn.query."),
            (".self_attn.k_proj.", ".attn.key."),
            (".self_attn.v_proj.", ".attn.value."),
            (".self_attn.out_proj.", ".attn.out."),
            (".encoder_attn.q_proj.", ".cross_attn.query."),
            (".encoder_attn.k_proj.", ".cross_attn.key."),
            (".encoder_attn.v_proj.", ".cross_attn.value."),
            (".encoder_attn.out_proj.", ".cross_attn.out."),
            ("decoder.embed_tokens.", "decoder.token_embedding."),
        ]

        var out: [String: MLXArray] = [:]
        for (k, v) in weights {
            var key = String(k.dropFirst("model.".count))
            var skip = false
            for (old, new) in keyMap {
                if key.contains(old) {
                    if let new {
                        key = key.replacingOccurrences(of: old, with: new)
                    } else {
                        skip = true
                    }
                    break
                }
            }
            if skip { continue }
            var value = v
            if key.contains("conv1.weight") || key.contains("conv2.weight") {
                if value.ndim == 3 {
                    value = value.transposed(0, 2, 1) // [O, C, K] -> [O, K, C]
                }
            }
            out[key] = value
        }
        return out
    }

    // MARK: - Transcribe

    func transcribe(audioSamples: [Float], language: String? = nil, maxTokens: Int = 200) async throws -> String {
        guard let model = self.model, let spec = self.spec, let tokenizer = self.tokenizer, let mel = self.mel else {
            throw WhisperMlxError.notLoaded
        }
        let sr = Float(WhisperMlxMel.sampleRate)
        let chunk = Int(Float(WhisperMlxMel.chunkSamples))
        var texts: [String] = []
        var offset = 0
        while offset < audioSamples.count || audioSamples.isEmpty {
            let window = Array(audioSamples[offset..<min(offset + chunk, audioSamples.count)])
            let text = try await Self.transcribeWindow(
                model: model, spec: spec, tokenizer: tokenizer, mel: mel,
                audio: window, language: language, maxTokens: maxTokens)
            if !text.isEmpty { texts.append(text) }
            offset += chunk
            if window.count < chunk { break }
        }
        return texts.joined(separator: " ").trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Whisper non-speech symbol strings (openai/whisper tokenizer.py +
    /// whisper.cpp non_speech_tokens). Decoded text that exactly matches one
    /// of these (or " " + token) is suppressed to avoid halluncinated speaker
    /// tags / music-note loops.
    private static let nonSpeechSymbols: [String] = [
        "\"", "#", "(", ")", "*", "+", "/", ":", ";", "<", "=", ">", "@",
        "[", "\\", "]", "^", "_", "`", "{", "|", "}", "~", "「", "」", "『",
        "』", "<<", ">>", "<<<", ">>>", "--", "---", "-(", "-[", "('", "(\"",
        "((", "))", "(((", ")))", "[[", "]]", "{{", "}}", "♪♪", "♪♪♪",
        "♩", "♪", "♫", "♬", "♭", "♮", "♯",
    ]

    private static func makeSuppressSet(
        tokenizer: WhisperMlxTokenizer, vocab: Int, spec: WhisperMlxSpec
    ) -> Set<Int> {
        // Suppress the whisper control tokens EXCEPT eot (50257): eot is the
        // end-of-text marker and must always stay sampleable, otherwise the
        // model keeps generating past the end of speech (repetition loop).
        var set = Set(50258...50363) // sot/langs/tasks/no_timestamps/specials
        set.insert(WhisperTokens.blank)
        // Timestamp tokens are never wanted (without_timestamps decode).
        let tsStart = 50364
        let tsEnd = min(tsStart + 3002, vocab)
        if tsStart < tsEnd {
            for t in tsStart..<tsEnd { set.insert(t) }
        }
        // Non-speech symbols: decode each vocab entry and suppress exact
        // matches (whisper.cpp suppress_nst with token and " " + token).
        for id in 0..<min(vocab, 50257) {
            let s = tokenizer.decode([id], skipSpecialTokens: true)
            if Self.nonSpeechSymbols.contains(s) {
                set.insert(id)
            }
        }
        return set
    }
    /// Per-window decode at a given temperature. Returns the decoded text plus
    /// the quality metrics the whisper fallback heuristic needs:
    ///   - avgLogprob: mean per-token log-probability of the generated tokens.
    ///   - compressionRatio: len(utf8) / len(zlib(utf8)); > 2.4 signals a loop.
    static func decodeWindow(
        model: WhisperMlxModel, spec: WhisperMlxSpec, tokenizer: WhisperMlxTokenizer,
        xa: MLXArray, promptTokens: [Int], temperature: Float, maxTokens: Int,
        suppress: Set<Int>
    ) -> (text: String, tokens: [Int], avgLogprob: Float, compressionRatio: Float) {
        let promptLen = promptTokens.count
        let prompt = MLXArray(promptTokens.map { Int32($0) }).reshaped([1, promptLen])
        var cache: [((MLXArray, MLXArray)?, (MLXArray, MLXArray)?)] = []
        let (logits0, cache0) = model.decode(prompt, xa: xa, kvCache: &cache, offset: 0)
        var logits = logits0
        cache = cache0
        var generated: [Int] = []
        var offsetT = promptLen
        var sumLogprob: Float = 0
        var sawNonBlank = false

        while generated.count < max(maxTokens - promptLen, 1) {
            // Sample the next token from the current logits (post-filter).
            let (next, logp) = Self.sampleNext(
                logits, generated: generated, temperature: temperature,
                suppress: suppress, vocab: spec.vocabSize,
                allowEOT: sawNonBlank)
            if next == WhisperTokens.eot, !sawNonBlank {
                // The model wants to stop before saying anything; treat the
                // audio as silence by stopping here (empty transcript).
                break
            }
            if next == WhisperTokens.eot { break }
            if next != WhisperTokens.blank { sawNonBlank = true }
            generated.append(next)
            sumLogprob += logp
            let tok = MLXArray([Int32(next)]).reshaped([1, 1])
            let (l, c) = model.decode(tok, xa: xa, kvCache: &cache, offset: offsetT)
            logits = l
            cache = c
            offsetT += 1
        }

        let avgLogprob: Float = generated.isEmpty ? 0 : sumLogprob / Float(generated.count)

        var text = tokenizer.decode(generated, skipSpecialTokens: true)
        text = text.trimmingCharacters(in: .whitespacesAndNewlines)
        // Whisper greedy decode artifacts: an opening quote that never closes,
        // or a stray leading/trailing quote. Strip balanced-looking quotes
        // while preserving inner punctuation.
        if text.count >= 2 {
            let first = text.first!
            let last = text.last!
            let isQuote = { (c: Character) -> Bool in
                ["\"", "\'", "\u{201C}", "\u{201D}", "\u{2018}", "\u{2019}", "`"].contains(c)
            }
            if isQuote(first) && isQuote(last) {
                text.removeFirst()
                text.removeLast()
            } else if isQuote(last) {
                text.removeLast()
            } else if isQuote(first) && text.count >= 2 {
                text.removeFirst()
            } else if isQuote(first) {
                text.removeFirst()
            }
            text = text.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        let ratio = Self.compressionRatio(of: text)
        return (text, generated, avgLogprob, ratio)
    }

    /// True when a decode result is too repetitive or too low-probability and
    /// should trigger the next temperature in the fallback chain (whisper
    /// `decode_with_fallback` heuristic).
    static func needsFallback(avgLogprob: Float, compressionRatio: Float, text: String) -> Bool {
        guard !text.isEmpty else { return false } // silence = valid empty result
        // Primary loop detector: repetitive text compresses extremely well.
        if compressionRatio > 2.4 { return true }
        // Completion / hallucination detector: whisper sometimes answers the
        // speaker with a bare punctuation mark or a one-token "reply" instead
        // of the transcript (e.g. a lone "?" / "。"). Such output carries very
        // little speech content and should trigger temperature fallback.
        if Self.isPunctuationOnly(text) { return true }
        return false
    }

    /// True when the decoded text carries no real words: only punctuation,
    /// quote-like characters, or a single short symbol.
    static func isPunctuationOnly(_ text: String) -> Bool {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return false }
        for ch in trimmed {
            if ch.isLetter || ch.isNumber { return false }
        }
        return trimmed.count <= 4
    }

    /// Rank decode results: prefer non-empty, lower compression (less looped),
    /// then higher average log-probability.
    static func isBetterResult(
        _ candidate: (text: String, avgLogprob: Float, compressionRatio: Float),
        than current: (text: String, avgLogprob: Float, compressionRatio: Float)?
    ) -> Bool {
        guard let current else { return !candidate.text.isEmpty }
        if candidate.text.isEmpty != current.text.isEmpty {
            return !candidate.text.isEmpty
        }
        if candidate.compressionRatio != current.compressionRatio {
            return candidate.compressionRatio < current.compressionRatio
        }
        return candidate.avgLogprob > current.avgLogprob
    }

    /// len(utf8) / len(zlib(utf8)) — high values indicate repetitive text.
    static func compressionRatio(of text: String) -> Float {
        let bytes = Array(text.utf8)
        guard !bytes.isEmpty else { return 0 }
        let compressed = try? (Data(bytes) as NSData).compressed(using: .zlib)
        let compLen = compressed?.count ?? bytes.count
        return Float(bytes.count) / Float(max(compLen, 1))
    }

    /// Select the next token by greedy argmax (T≈0) or categorical sampling
    /// (T>0, logits/T) with the full whisper logit suppression applied.
    /// Returns (token, log-probability of the token under the filtered dist).
    private static func sampleNext(
        _ logits: MLXArray, generated: [Int], temperature: Float,
        suppress: Set<Int>, vocab: Int, allowEOT: Bool
    ) -> (Int, Float) {
        let last = logits[0, logits.shape[1] - 1, 0...]
        var mask = [Float](repeating: 0, count: vocab)
        for t in suppress where t < vocab {
            mask[t] = -Float.infinity
        }
        if !allowEOT {
            // Suppress the end-of-text token until something non-blank has been
            // generated: with a silence-padded window, greedy whisper prefers
            // EOT immediately, which would skip real speech that follows.
            mask[WhisperTokens.eot] = -Float.infinity
        }
        let s = (last + MLXArray(mask).asType(last.dtype)).asType(.float32)

        let token: Int
        if temperature > 0.01 {
            let adjusted = s / temperature
            token = Int(MLXRandom.categorical(adjusted).item(Int32.self))
        } else {
            token = Int(argMax(s).item(Int32.self))
        }
        // Log-probability of the sampled token under softmax(s):
        // logp = s[token] - logsumexp(s). Vectorized over the vocab.
        // (s is 1-D [vocab]; logSumExp without keepDims yields a 0-d scalar.)
        let logsumexp = s.logSumExp(axis: -1)
        let logp = (s[token] - logsumexp).item(Float.self)
        return (token, logp.isNaN ? -20 : logp)
    }

    private static func transcribeWindow(
        model: WhisperMlxModel, spec: WhisperMlxSpec, tokenizer: WhisperMlxTokenizer,
        mel: WhisperMlxMel, audio: [Float], language: String?, maxTokens: Int
    ) async throws -> String {
        guard let feats = mel.compute(audio: audio) else { return "" }
        let xa = model.encode(feats) // [1, 1500, state]

        var tokens: [Int] = [WhisperTokens.sot]
        var resolvedLang: String?
        if spec.vocabSize >= 51865 {
            if let lang = language, !lang.isEmpty, lang != "auto" {
                resolvedLang = lang
            } else {
                // Detect language from the logits of the SOT-only prefill.
                var detectCache: [((MLXArray, MLXArray)?, (MLXArray, MLXArray)?)] = []
                let (detectLogits, _) = model.decode(
                    MLXArray([Int32(WhisperTokens.sot)]).reshaped([1, 1]),
                    xa: xa, kvCache: &detectCache, offset: 0)
                let last = detectLogits[0, 0, 0...]
                var best = 0
                var bestVal = Float.leastNormalMagnitude
                for (i, code) in WhisperTokens.languages.enumerated() {
                    let v = last[WhisperTokens.languageToken(code)].item(Float.self)
                    if v > bestVal { bestVal = v; best = i }
                }
                resolvedLang = WhisperTokens.languages[best]
            }
            if let lang = resolvedLang {
                tokens.append(WhisperTokens.languageToken(lang))
            }
            tokens.append(WhisperTokens.transcribe)
            tokens.append(WhisperTokens.noTimestamps)
        }

        let suppress = Self.makeSuppressSet(tokenizer: tokenizer, vocab: spec.vocabSize, spec: spec)

        // Temperature fallback chain (whisper decode_with_fallback):
        // greedy first, then progressively noisier sampling when the greedy
        // result looks like a loop (high compression ratio) or hallucination
        // (very low average log-probability).
        let temperatures: [Float] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        var best: (text: String, avgLogprob: Float, compressionRatio: Float)?
        for t in temperatures {
            let result = Self.decodeWindow(
                model: model, spec: spec, tokenizer: tokenizer, xa: xa,
                promptTokens: tokens, temperature: t, maxTokens: maxTokens,
                suppress: suppress)
            let candidate = (result.text, result.avgLogprob, result.compressionRatio)
            if Self.isBetterResult(candidate, than: best) {
                best = candidate
            }
            if !Self.needsFallback(avgLogprob: result.avgLogprob, compressionRatio: result.compressionRatio, text: result.text) {
                return result.text
            }
            // Result looks looped (high compression); retry at higher temperature.
        }

        // All temperatures looped or empty — return the best sampled result.
        return best?.text ?? ""
    }
}
