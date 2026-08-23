import Foundation
import MLX

// MARK: - Parakeet TDT MLX engine (app-managed lifecycle, catalog-driven)

actor ParakeetMlxEngine {
    private var model: ParakeetMlxModel?
    private var spec: ParakeetMlxSpec?
    private var currentDirectory: URL?

    static func modelsRootDirectory() -> URL {
        Qwen3MlxEngine.modelsRootDirectory()
    }

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
            throw ParakeetMlxError.invalidWeights("Unknown Parakeet card \(modelID)/\(variantID)")
        }
        let directory = cacheDirectory(modelID: modelID, variantID: variantID)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let fm = FileManager.default
        let base = SettingsStore.shared.huggingFaceBaseURL
        let repoName = String(card.repo.split(separator: "/").last ?? "")

        func fetchData(_ relativePath: String) async throws -> Data {
            guard let url = URL(string: "\(base)/\(card.repo)/resolve/main/\(relativePath)") else {
                throw URLError(.badURL)
            }
            var request = URLRequest(url: url)
            request.timeoutInterval = 30
            let (data, _) = try await URLSession.shared.data(for: request)
            return data
        }

        for file in ["config.json", "tokenizer.model", "tokenizer.vocab", "vocab.txt"] {
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
            throw ParakeetMlxError.invalidWeights("Bad config.json")
        }
        let spec = try ParakeetMlxSpec.parse(from: config)

        let weightsURL = dir.appendingPathComponent("model.safetensors")
        let arrays = try MLX.loadArrays(url: weightsURL)

        // All weights cast to bf16 (matches parakeet-mlx from_pretrained)
        var w: [String: MLXArray] = [:]
        for (k, v) in arrays {
            w[k] = v.asType(.bfloat16)
        }

        let model = ParakeetMlxModel(spec: spec, w: w)
        self.model = model
        self.spec = spec
        self.currentDirectory = dir
    }

    // MARK: - Transcribe (TDT greedy)

    func transcribe(audioSamples: [Float], maxSymbols: Int? = nil) async throws -> String {
        guard let model = self.model, let spec = self.spec else {
            throw ParakeetMlxError.notLoaded
        }

        guard let melArray = ParakeetMlxMelSpectrogram(spec: spec).computeMLX(audio: audioSamples) else {
            throw ParakeetMlxError.invalidAudio
        }
        let t = melArray.shape[1]
        let (features, lengths) = model.encoder(melArray, lengths: MLXArray([t]))
        let total = min(Int(features.shape[1]), 8_192)
        let blankID = spec.decoderVocabSize
        let maxSym = maxSymbols ?? spec.maxSymbols

        var tokens: [Int] = []
        var step = 0
        var lastToken: Int? = nil
        var hidden: [MLXArray?] = []
        var cell: [MLXArray?] = []
        var newSymbols = 0

        while step < total {
            let (decOut, h, c) = model.decoder(y: lastToken, hidden: hidden, cell: cell)

            let encStep = features[0..., step..<(step + 1), 0...] // [1, 1, 1024]
            let jointOut = model.joint(encStep, decOut) // [1, 1, 1, 8198]

            let tokenLogits = jointOut[0, 0, 0, 0..<blankID]
            let predToken = Int(argMax(tokenLogits).item(Int32.self))
            let durationLogits = jointOut[0, 0, 0, blankID...]
            let decision = Int(argMax(durationLogits).item(Int32.self))
            let dur = spec.durations[min(max(0, decision), spec.durations.count - 1)]

            if predToken != blankID {
                tokens.append(predToken)
                lastToken = predToken
                hidden = h.map { Optional($0) }
                cell = c.map { Optional($0) }
            }

            step += dur
            newSymbols += 1
            if dur != 0 {
                newSymbols = 0
            } else if maxSym > 0, newSymbols >= maxSym {
                step += 1
                newSymbols = 0
            }
        }

        return spec.decode(tokens: tokens)
    }
}
