import Foundation
import MLX

// MARK: - Fun-ASR MLX engine (app-managed lifecycle, catalog-driven)

actor FunMlxEngine {
    private var model: FunMlxModel?
    private var spec: FunMlxSpec?
    private var tokenizer: MlxSttTokenizerJson?
    private var frontend: FunMlxFrontend?
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
            throw FunMlxError.invalidWeights("Unknown Fun card \(modelID)/\(variantID)")
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

        for file in ["config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"] {
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
            throw FunMlxError.invalidWeights("Bad config.json")
        }
        let spec = try FunMlxSpec.parse(from: config)
        let quant = Qwen3MlxQuantInfo.parse(from: config)

        let weightsURL = dir.appendingPathComponent("model.safetensors")
        let arrays = try MLX.loadArrays(url: weightsURL)
        var w: [String: MLXArray] = [:]
        for (k, v) in arrays { w[k] = v }

        let model = FunMlxModel(spec: spec, w: w, quant: quant)
        let tokURL = dir.appendingPathComponent("tokenizer.json")
        let tokenizer = try MlxSttTokenizerJson(contentsOf: tokURL)

        self.model = model
        self.spec = spec
        self.tokenizer = tokenizer
        self.frontend = FunMlxFrontend(spec: spec)
        self.currentDirectory = dir
    }

    // MARK: - Transcribe

    func transcribe(audioSamples: [Float], maxNewTokens: Int = 256) async throws -> String {
        guard let model = self.model, let spec = self.spec, let tokenizer = self.tokenizer, let frontend = self.frontend else {
            throw FunMlxError.notLoaded
        }

        guard let (feats, speechLen) = frontend.compute(audio: audioSamples) else {
            throw FunMlxError.invalidAudio
        }
        let audioEncoded = model.encoder(feats, ilens: speechLen)
        let (adaptorOut, _) = model.adaptor(audioEncoded, ilens: speechLen)

        // Prompt (mlx-audio-plus funasr._prepare_prompt). The audio embeddings
        // are spliced in where `<|startofspeech|><|endofspeech|>` sits. In this
        // tokenizer those two are NOT added tokens (they BPE into multi-token
        // sequences that both start with the same id), so the reference finds
        // the first occurrence of that id and inserts the audio embeddings
        // right after it, keeping the surrounding text embeddings.
        let systemPrompt =
            "You are a speech recognition assistant. Transcribe the audio accurately. "
            + "Output only the transcription, nothing else."
        let prompt = "<|im_start|>system\n\(systemPrompt)<|im_end|>"
            + "<|im_start|>user\n<|startofspeech|><|endofspeech|>"
            + "<|im_end|><|im_start|>assistant\n"
        let ids = tokenizer.encode(prompt)
        let sosID = tokenizer.encode("<|startofspeech|>").first ?? -1
        let eosID = tokenizer.encode("<|endofspeech|>").first ?? -1
        let sosPos = ids.firstIndex(of: sosID) ?? ids.count - 1
        let eosPos = ids.firstIndex(of: eosID) ?? sosPos

        let idArray = MLXArray(ids.map { Int32($0) }).reshaped([1, ids.count])
        let textEmb = model.llm.embedTokens(idArray) // [1, T, hidden]
        let headEmb = textEmb[0..., 0..<(sosPos + 1), 0...] // includes sos marker
        let tailEmb = textEmb[0..., eosPos..., 0...] // includes eos marker onward
        let speech = adaptorOut[0..., 0..., 0...].asType(textEmb.dtype)
        // Run the LLM in float32: the merged features carry huge audio-domain
        // values and fp16 overflows inside the quantized MLPs (NaN by layer 3).
        let embeddings = concatenated([headEmb, speech, tailEmb], axis: 1).asType(.float32)

        let cache = Qwen3MlxKVCache()
        var logits = model.llm(embeddings, cache: cache, isEmbeds: true)
        cache.offset = embeddings.shape[1]

        // EOS set mirrors mlx-audio-plus _setup_special_tokens.
        var eosTokens = Set<Int>()
        for t in ["<|im_end|>", "<|endoftext|>", "</s>", "<|endofspeech|>"] {
            if let first = tokenizer.encode(t).first { eosTokens.insert(first) }
        }

        var generated: [Int] = []
        var current = Int(argMax(logits[0, logits.shape[1] - 1, 0...]).item(Int32.self))
        for _ in 0..<maxNewTokens {
            if eosTokens.contains(current) { break }
            generated.append(current)
            // Decode embeds come from the fp16 table; raise to fp32 so the
            // layer math stays in the same precision as the prefill.
            let tokenEmbed = model.llm.embedTokens(MLXArray([Int32(current)]).reshaped([1, 1])).asType(.float32)
            logits = model.llm(tokenEmbed, cache: cache, isEmbeds: true)
            cache.offset += 1
            current = Int(argMax(logits[0, logits.shape[1] - 1, 0...]).item(Int32.self))
        }

        let text = tokenizer.decode(generated, skipSpecialTokens: true)
        return text.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
