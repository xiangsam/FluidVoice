import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - Qwen3-ASR MLX Engine (fully in-app, app-managed lifecycle, catalog-driven)

actor Qwen3MlxEngine {
    private var encoderModel: Qwen3MlxAudioEncoder?
    private var decoderModel: Qwen3MlxTextDecoder?
    private var spec: Qwen3MlxSpec?
    private var loadedVariantID: String?
    private var currentDirectory: URL?

    static func modelsRootDirectory() -> URL {
        if let appSupport = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask
        ).first {
            return
                appSupport
                .appendingPathComponent("FluidVoice", isDirectory: true)
                .appendingPathComponent("Models", isDirectory: true)
        }
        return FileManager.default.temporaryDirectory
            .appendingPathComponent("FluidVoice", isDirectory: true)
            .appendingPathComponent("Models", isDirectory: true)
    }

    /// Cache directory for one variant: .../Models/MlxStt/<modelID>/<variantID>/
    static func cacheDirectory(modelID: String, variantID: String) -> URL {
        modelsRootDirectory()
            .appendingPathComponent("MlxStt", isDirectory: true)
            .appendingPathComponent(modelID, isDirectory: true)
            .appendingPathComponent(variantID, isDirectory: true)
    }

    static func modelsExist(modelID: String, variantID: String) -> Bool {
        guard MlxSttCatalog.card(pathID: "\(modelID)/\(variantID)") != nil else {
            return false
        }
        let dir = cacheDirectory(modelID: modelID, variantID: variantID)
        guard
            FileManager.default.fileExists(
                atPath: dir.appendingPathComponent("model.safetensors.index.json").path)
        else {
            return false
        }
        guard
            let data = try? Data(
                contentsOf: dir.appendingPathComponent("model.safetensors.index.json")),
            let index = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let weightMap = index["weight_map"] as? [String: String]
        else {
            return false
        }
        return Set(weightMap.values).allSatisfy { file in
            FileManager.default.fileExists(atPath: dir.appendingPathComponent(file).path)
        }
    }

    // MARK: - Download (managed by the app, mirror-aware)

    @discardableResult
    static func download(
        modelID: String, variantID: String,
        progressHandler: ((Double) -> Void)? = nil
    ) async throws -> URL {
        guard let card = MlxSttCatalog.card(pathID: "\(modelID)/\(variantID)") else {
            throw Qwen3MlxError.invalidWeights("Unknown MLX STT variant \(modelID)/\(variantID)")
        }
        let directory = cacheDirectory(modelID: modelID, variantID: variantID)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let fm = FileManager.default
        let base = SettingsStore.shared.huggingFaceBaseURL

        func fetchData(_ relativePath: String, owner: String, repo: String) async throws -> Data {
            guard
                let url = URL(string: "\(base)/\(owner)/\(repo)/resolve/main/\(relativePath)")
            else {
                throw URLError(.badURL)
            }
            var request = URLRequest(url: url)
            request.timeoutInterval = 30
            let (data, _) = try await URLSession.shared.data(for: request)
            return data
        }

        // 1. config.json (+ tokenizer pieces from the official Qwen repo)
        let configDest = directory.appendingPathComponent("config.json")
        if !fm.fileExists(atPath: configDest.path) {
            let data = try await fetchData(
                "config.json", owner: "mlx-community", repo: String(card.repo.split(separator: "/").last ?? ""))
            try data.write(to: configDest)
        }
        for file in ["vocab.json", "merges.txt"] {
            let dest = directory.appendingPathComponent(file)
            if !fm.fileExists(atPath: dest.path) {
                let data = try await fetchData(file, owner: "Qwen", repo: "Qwen3-ASR-0.6B")
                try data.write(to: dest)
            }
        }

        // 2. index.json
        let indexURL = directory.appendingPathComponent("model.safetensors.index.json")
        if !fm.fileExists(atPath: indexURL.path) {
            let data = try await fetchData(
                "model.safetensors.index.json",
                owner: "mlx-community",
                repo: String(card.repo.split(separator: "/").last ?? ""))
            try data.write(to: indexURL)
        }

        // 3. shards listed in the index
        guard
            let data = try? Data(contentsOf: indexURL),
            let index = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let weightMap = index["weight_map"] as? [String: String]
        else {
            throw Qwen3MlxError.invalidWeights("Missing safetensors index")
        }
        let shards = Set(weightMap.values).sorted()
        let repoName = String(card.repo.split(separator: "/").last ?? "")
        var completed = 0.0
        let total = Double(max(1, shards.count))
        for shard in shards {
            let dest = directory.appendingPathComponent(shard)
            try Task.checkCancellation()
            if !fm.fileExists(atPath: dest.path) || (try? fm.attributesOfItem(atPath: dest.path)[.size] as? Int) == nil || ((try? fm.attributesOfItem(atPath: dest.path)[.size] as? Int) ?? 0) < 1024 {
                guard let url = URL(string: "\(base)/mlx-community/\(repoName)/resolve/main/\(shard)") else {
                    throw URLError(.badURL)
                }
                try await MlxModelDownloader.downloadFile(
                    from: url,
                    to: dest,
                    progress: { fraction in
                        Task { @MainActor in
                            progressHandler?((Double(completed) + fraction) / total)
                        }
                    }
                )
            }
            completed += 1
            progressHandler?(completed / total)
        }
        return directory
    }

    // MARK: - Load / unload

    func loadModels(
        modelID: String, variantID: String,
        directory: URL? = nil
    ) async throws {
        let dir = directory ?? Self.cacheDirectory(modelID: modelID, variantID: variantID)

        // 1. Parse config.json -> spec + quantization info
        let configData = try Data(contentsOf: dir.appendingPathComponent("config.json"))
        guard
            let config = try JSONSerialization.jsonObject(with: configData) as? [String: Any]
        else {
            throw Qwen3MlxError.invalidWeights("Bad config.json")
        }
        let spec = try Qwen3MlxSpec.parse(from: config)
        let quant = Qwen3MlxQuantInfo.parse(from: config)

        // 2. Load all safetensors shards and merge
        let indexData = try Data(
            contentsOf: dir.appendingPathComponent("model.safetensors.index.json"))
        guard
            let index = try JSONSerialization.jsonObject(with: indexData) as? [String: Any],
            let weightMap = index["weight_map"] as? [String: String]
        else {
            throw Qwen3MlxError.invalidWeights("Bad safetensors index")
        }
        var flatWeights: [String: MLXArray] = [:]
        let shardFiles = Set(weightMap.values).sorted()
        for shard in shardFiles {
            let url = dir.appendingPathComponent(shard)
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw Qwen3MlxError.invalidWeights("Missing shard \(shard)")
            }
            let arrays = try MLX.loadArrays(url: url)
            for (k, v) in arrays {
                flatWeights[k] = v
            }
        }

        // 3. Build models from spec + weights (construction-time injection)
        var encoderW: [String: MLXArray] = [:]
        var decoderW: [String: MLXArray] = [:]
        for (key, value) in flatWeights {
            if key.contains("positional") { continue }
            if key.hasPrefix("audio_tower.") {
                encoderW[String(key.dropFirst("audio_tower.".count))] = value
            } else if key.hasPrefix("model.") {
                decoderW[String(key.dropFirst("model.".count))] = value
            }
        }
        let encoder = Qwen3MlxAudioEncoder(spec: spec, w: encoderW, quant: quant)
        let decoder = Qwen3MlxTextDecoder(spec: spec, w: decoderW, quant: quant)

        self.encoderModel = encoder
        self.decoderModel = decoder
        self.spec = spec
        self.loadedVariantID = variantID
        self.currentDirectory = dir
    }

    // MARK: - Transcribe

    func transcribe(
        audioSamples: [Float],
        language isoCode: String? = nil,
        maxNewTokens: Int = 256,
        contextWords: String? = nil
    ) async throws -> String {
        guard let encoder = self.encoderModel, let decoder = self.decoderModel else {
            throw Qwen3MlxError.notLoaded
        }

        let mel = Qwen3MlxMelSpectrogram().compute(audio: audioSamples)
        guard !mel.isEmpty, let first = mel.first, !first.isEmpty else {
            throw Qwen3MlxError.invalidAudio
        }
        let nMel = mel.count
        let tLen = first.count

        var melData = [Float](repeating: 0, count: nMel * tLen)
        for m in 0..<nMel {
            for t in 0..<tLen {
                melData[m * tLen + t] = mel[m][t]
            }
        }
        let melArray = MLXArray(melData, [nMel, tLen])

        let audioFeatures = encoder(melArray)

        let directory = self.currentDirectory
            ?? Self.cacheDirectory(modelID: "qwen3-asr", variantID: "0.6B-8bit")

        // Vocabulary line for context/hotwords
        var contextTokens: [Int] = []
        if let words = contextWords, !words.isEmpty {
            contextTokens = try Self.tokenize("Vocabulary: \(words)", directory: directory)
        }

        // Language line (assistant turn): "language {name}<asr_text>"
        var languageTokens: [Int] = []
        if let isoCode, !isoCode.isEmpty, isoCode.lowercased() != "auto" {
            languageTokens = try Self.tokenize(Qwen3MlxLanguage.name(for: isoCode), directory: directory)
        }

        let nAudio = audioFeatures.shape[1]
        let prompt = Qwen3MlxBuilder.buildPrompt(
            nAudioTokens: nAudio, contextTokens: contextTokens, languageTokens: languageTokens)

        // Inject audio features at pad positions
        let embeddings = Self.prepareInputs(
            encoderOutput: audioFeatures, inputIds: prompt, decoder: decoder)

        // Generate
        let cache = Qwen3MlxKVCache()
        var logits = decoder(embeddings, cache: cache, isEmbeds: true)
        eval(logits)
        cache.offset = prompt.count

        var generated: [Int] = []
        var next = Self.argmax(logits[0, logits.shape[1] - 1, 0...])
        for _ in 0..<maxNewTokens {
            if Qwen3MlxTokens.eosTokenIds.contains(next) { break }
            generated.append(next)
            let tokenEmbed = decoder.embedTokens(MLXArray([next]).reshaped([1, 1]))
            logits = decoder(tokenEmbed, cache: cache, isEmbeds: true)
            eval(logits)
            cache.offset += 1
            next = Self.argmax(logits[0, logits.shape[1] - 1, 0...])
        }

        // Decode tokens to text
        let vocabulary = try Self.vocabularyTable(directory: directory)
        return Self.decodeTokens(generated, vocabulary: vocabulary)
    }

    private static func prepareInputs(
        encoderOutput: MLXArray, inputIds: [Int], decoder: Qwen3MlxTextDecoder
    ) -> MLXArray {
        let ids = MLXArray(inputIds)
        let embeddings = decoder.embedTokens(ids[.newAxis, 0...])

        let audioPositions = inputIds.enumerated().compactMap {
            $0.element == Qwen3MlxTokens.audioPadTokenId ? $0.offset : nil
        }
        guard !audioPositions.isEmpty else { return embeddings }
        let nAudio = encoderOutput.shape[1]
        let t = embeddings.shape[1]
        let h = embeddings.shape[2]

        // Materialize encoder features and splice them in at pad positions.
        let encVals = encoderOutput.reshaped([nAudio, h]).asArray(Float.self)
        var audioRows = [Float](repeating: 0, count: t * h)
        for (i, pos) in audioPositions.prefix(nAudio).enumerated() {
            for j in 0..<h {
                audioRows[pos * h + j] = encVals[i * h + j]
            }
        }
        let audioFull = MLXArray(audioRows, [1, t, h]).asType(embeddings.dtype)

        var maskData = [Float](repeating: 0, count: t)
        for pos in audioPositions { maskData[pos] = 1 }
        let mask = MLXArray(maskData).reshaped([1, t, 1])
        return which(mask, audioFull, embeddings)
    }

    private static func argmax(_ logits: MLXArray) -> Int {
        Int(argMax(logits).item(Int32.self))
    }

    // MARK: - Tokenizer helpers (cached model directory)

    private static func vocabularyTable(directory: URL) throws -> [Int: String] {
        let data = try Data(contentsOf: directory.appendingPathComponent("vocab.json"))
        guard let stringToId = try JSONSerialization.jsonObject(with: data) as? [String: Int] else {
            throw Qwen3MlxError.invalidWeights("Bad vocab.json")
        }
        var idToString: [Int: String] = [:]
        for (token, id) in stringToId { idToString[id] = token }
        return idToString
    }

    private static func tokenize(_ text: String, directory: URL) throws -> [Int] {
        let data = try Data(contentsOf: directory.appendingPathComponent("vocab.json"))
        guard let vocab = try JSONSerialization.jsonObject(with: data) as? [String: Int],
            let mergesData = try? Data(contentsOf: directory.appendingPathComponent("merges.txt")),
            let mergesText = String(data: mergesData, encoding: .utf8)
        else {
            throw Qwen3MlxError.invalidWeights("Missing vocab.json/merges.txt")
        }
        var merges: [String: Int] = [:]
        for line in mergesText.split(separator: "\n") {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.isEmpty || trimmed.hasPrefix("#") { continue }
            let parts = trimmed.split(separator: " ", maxSplits: 1)
            guard parts.count == 2 else { continue }
            merges["\(parts[0]) \(parts[1])"] = merges.count
        }
        return encodeBPE(text, vocabToID: vocab, merges: merges)
    }

    // MARK: - BPE (official gpt2-style, verified against merges.txt)

    private static let byteEncoder: [UInt8: UInt32] = {
        var bs = [Int]()
        bs.append(contentsOf: 33...126)
        bs.append(contentsOf: 161...172)
        bs.append(contentsOf: 174...255)
        var cs = bs
        var n = 0
        for b in 0...255 where !bs.contains(b) {
            bs.append(b)
            cs.append(256 + n)
            n += 1
        }
        return Dictionary(uniqueKeysWithValues: zip(bs.map { UInt8($0) }, cs.map { UInt32($0) }))
    }()

    private static func encodeBPE(_ text: String, vocabToID: [String: Int], merges: [String: Int]) -> [Int] {
        let chars = Array(text.utf8).map { byte -> String in
            let cp = byteEncoder[byte] ?? UInt32(byte)
            guard let scalar = UnicodeScalar(cp) else { return "" }
            return String(Character(scalar))
        }
        guard !chars.isEmpty else { return [] }
        var segments = chars
        while segments.count > 1 {
            var bestRank = Int.max
            var bestIndex = -1
            for i in 0..<(segments.count - 1) {
                let key = "\(segments[i]) \(segments[i + 1])"
                if let rank = merges[key], rank < bestRank {
                    bestRank = rank
                    bestIndex = i
                }
            }
            guard bestIndex >= 0 else { break }
            segments[bestIndex] += segments[bestIndex + 1]
            segments.remove(at: bestIndex + 1)
        }
        return segments.compactMap { vocabToID[$0] }
    }

    // MARK: - Output decoding

    /// Inverse of `byteEncoder`: gpt2 byte-encoded codepoint -> original byte.
    private static let byteDecoder: [UInt32: UInt8] = {
        var d: [UInt32: UInt8] = [:]
        for (b, cp) in byteEncoder { d[cp] = b }
        return d
    }()

    private static func decodeTokens(_ tokenIds: [Int], vocabulary: [Int: String]) -> String {
        var startIdx = 0
        if let asrIdx = tokenIds.firstIndex(of: Qwen3MlxTokens.asrTextTokenId) {
            startIdx = asrIdx + 1
        }
        let transcriptionTokens = Array(tokenIds[startIdx...])
        var pieces: [String] = []
        for id in transcriptionTokens {
            if let piece = vocabulary[id] {
                pieces.append(piece)
            }
        }
        let raw = pieces.joined().replacingOccurrences(of: "<|im_end|>", with: "")

        var bytes = [UInt8]()
        for scalar in raw.unicodeScalars {
            if let b = byteDecoder[scalar.value] {
                bytes.append(b)
            } else if scalar.value < 256 {
                bytes.append(UInt8(scalar.value))
            }
        }
        let decoded = String(bytes: bytes, encoding: .utf8) ?? raw
        return decoded.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

// MARK: - Prompt builder (mirrors qwen3-asr-mlx build_prompt)

enum Qwen3MlxBuilder {
    static func buildPrompt(
        nAudioTokens: Int, contextTokens: [Int], languageTokens: [Int]
    ) -> [Int] {
        var prompt: [Int] = [
            Qwen3MlxTokens.imStartTokenId,
            Qwen3MlxTokens.systemTokenId,
            Qwen3MlxTokens.newlineTokenId,
        ]
        if !contextTokens.isEmpty {
            prompt.append(contentsOf: contextTokens)
        }
        prompt += [
            Qwen3MlxTokens.imEndTokenId,
            Qwen3MlxTokens.newlineTokenId,
            Qwen3MlxTokens.imStartTokenId,
            Qwen3MlxTokens.userTokenId,
            Qwen3MlxTokens.newlineTokenId,
            Qwen3MlxTokens.audioStartTokenId,
        ]
        prompt += [Int](repeating: Qwen3MlxTokens.audioPadTokenId, count: nAudioTokens)
        prompt += [
            Qwen3MlxTokens.audioEndTokenId,
            Qwen3MlxTokens.imEndTokenId,
            Qwen3MlxTokens.newlineTokenId,
            Qwen3MlxTokens.imStartTokenId,
            Qwen3MlxTokens.assistantTokenId,
            Qwen3MlxTokens.newlineTokenId,
        ]
        if !languageTokens.isEmpty {
            prompt.append(Qwen3MlxTokens.languageTokenId)
            prompt.append(contentsOf: languageTokens)
            prompt.append(Qwen3MlxTokens.asrTextTokenId)
        }
        return prompt
    }
}

// MARK: - Language name mapping

enum Qwen3MlxLanguage {
    static func name(for isoCode: String) -> String {
        switch isoCode.lowercased() {
        case "zh", "zh-cn", "chinese": return "Chinese"
        case "en", "english": return "English"
        case "ja", "japanese": return "Japanese"
        case "ko", "korean": return "Korean"
        case "yue", "cantonese": return "Cantonese"
        case "vi", "vietnamese": return "Vietnamese"
        case "th", "thai": return "Thai"
        default: return "English"
        }
    }
}

// MARK: - Errors

enum Qwen3MlxError: Error, LocalizedError {
    case notLoaded
    case invalidAudio
    case invalidWeights(String)

    var errorDescription: String? {
        switch self {
        case .notLoaded: return "Qwen3 MLX model not loaded"
        case .invalidAudio: return "Audio too short for Qwen3-ASR"
        case .invalidWeights(let detail): return "Qwen3 MLX weights invalid: \(detail)"
        }
    }
}
