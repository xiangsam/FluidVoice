import Foundation
import MLX

// MARK: - GLM-ASR MLX engine (app-managed lifecycle, catalog-driven)

actor GlmMlxEngine {
    private var model: GlmMlxModel?
    private var spec: GlmMlxSpec?
    private var tokenizer: MlxSttTokenizerJson?
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
            size > 1_000_000_000
        else { return false }
        return FileManager.default.fileExists(
            atPath: dir.appendingPathComponent("config.json").path)
    }

    // MARK: - Download

    static func download(
        modelID: String, variantID: String,
        progressHandler: ((Double) -> Void)? = nil
    ) async throws -> URL {
        guard let card = MlxSttCatalog.card(pathID: "\(modelID)/\(variantID)") else {
            throw GlmMlxError.invalidWeights("Unknown GLM card \(modelID)/\(variantID)")
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
            request.timeoutInterval = 30
            let (data, _) = try await URLSession.shared.data(for: request)
            return data
        }

        for file in ["config.json", "tokenizer.json", "tokenizer_config.json"] {
            let dest = directory.appendingPathComponent(file)
            if !fm.fileExists(atPath: dest.path) {
                let data = try await fetchData(file)
                try data.write(to: dest)
            }
        }

        let weightsDest = directory.appendingPathComponent("model.safetensors")
        if !fm.fileExists(atPath: weightsDest.path)
            || ((try? (fm.attributesOfItem(atPath: weightsDest.path)[.size] as? Int) ?? 0) ?? 0) < 1_000_000_000
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

    // MARK: - Load

    func loadModels(modelID: String, variantID: String, directory: URL? = nil) async throws {
        let dir = directory ?? Self.cacheDirectory(modelID: modelID, variantID: variantID)
        let configData = try Data(contentsOf: dir.appendingPathComponent("config.json"))
        guard let config = try JSONSerialization.jsonObject(with: configData) as? [String: Any] else {
            throw GlmMlxError.invalidWeights("Bad config.json")
        }
        let spec = try GlmMlxSpec.parse(from: config)
        let quant = Qwen3MlxQuantInfo.parse(from: config)

        let weightsURL = dir.appendingPathComponent("model.safetensors")
        let arrays = try MLX.loadArrays(url: weightsURL)
        var w: [String: MLXArray] = [:]
        for (k, v) in arrays { w[k] = v }

        let model = GlmMlxModel(spec: spec, w: w, quant: quant)
        let tokURL = dir.appendingPathComponent("tokenizer.json")
        let tokenizer = try MlxSttTokenizerJson(contentsOf: tokURL)

        self.model = model
        self.spec = spec
        self.tokenizer = tokenizer
        self.currentDirectory = dir
    }

    // MARK: - Transcribe

    func transcribe(audioSamples: [Float], maxNewTokens: Int = 256) async throws -> String {
        guard let model = self.model, let spec = self.spec, let tokenizer = self.tokenizer else {
            throw GlmMlxError.notLoaded
        }

        let mel = GlmMlxMelSpectrogram(spec: spec).computeMLX(audio: audioSamples)
        guard let mel else { throw GlmMlxError.invalidAudio }
        let (audioEmbeds, audioLen) = model.encodeAudio(mel)

        // Prompt: <|user|>\n<|begin_of_audio|>[pad x N]<|end_of_audio|>\nPlease transcribe...<|assistant|>\n
        let headText = "<|user|>\n<|begin_of_audio|>"
        let tailText =
            "<|end_of_audio|>\nPlease transcribe this audio into text<|assistant|>\n"
        let headTokens = tokenizer.encode(headText)
        let tailTokens = tokenizer.encode(tailText)
        var prompt = headTokens
        prompt += [Int](repeating: 0, count: audioLen)
        prompt += tailTokens
        let audioStart = headTokens.count
        let t = prompt.count

        // Merge audio embeddings into text embeddings at placeholder positions.
        let ids = MLXArray(prompt)
        let embeddings0 = model.embedTokens(ids[.newAxis, 0...])
        let headEmbeds = embeddings0[0..., 0..<audioStart, 0...]
        let tailEmbeds = embeddings0[0..., (audioStart + audioLen)..., 0...]
        let embeddings = concatenated([headEmbeds, audioEmbeds, tailEmbeds], axis: 1)

        // Prefill
        let cache = Qwen3MlxKVCache()
        var logits = model.llama(embeddings, cache: cache, isEmbeds: true)
        cache.offset = t

        var generated: [Int] = []
        var current = Int(argMax(logits[0, logits.shape[1] - 1, 0...]).item(Int32.self))
        for _ in 0..<maxNewTokens {
            if spec.eosTokenIds.contains(current) { break }
            generated.append(current)
            let tokenEmbed = model.embedTokens(MLXArray([Int32(current)]).reshaped([1, 1]))
            logits = model.llama(tokenEmbed, cache: cache, isEmbeds: true)
            cache.offset += 1
            current = Int(argMax(logits[0, logits.shape[1] - 1, 0...]).item(Int32.self))
        }

        let text = tokenizer.decode(generated, skipSpecialTokens: true)
        return text.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
