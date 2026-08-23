import Foundation
import MLX

// MARK: - Nemotron 3.5 ASR MLX engine (app-managed lifecycle, catalog-driven)

actor NemotronMlxEngine {
    private var model: NemotronMlxModel?
    private var spec: NemotronMlxSpec?
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
            throw NemotronMlxError.invalidWeights("Unknown Nemotron card \(modelID)/\(variantID)")
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

        for file in ["config.json", "tokenizer.model", "vocab.txt"] {
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
            throw NemotronMlxError.invalidWeights("Bad config.json")
        }
        let spec = try Self.parseSpec(from: config)

        let weightsURL = dir.appendingPathComponent("model.safetensors")
        let arrays = try MLX.loadArrays(url: weightsURL)

        var w: [String: MLXArray] = [:]
        for (k, v) in arrays {
            w[k] = v
        }

        let model = NemotronMlxModel(spec: spec, w: w)
        self.model = model
        self.spec = spec
        self.currentDirectory = dir
    }

    static func parseSpec(from config: [String: Any]) throws -> NemotronMlxSpec {
        func int(_ d: [String: Any]?, _ k: String, _ def: Int) -> Int {
            (d?[k] as? NSNumber)?.intValue ?? def
        }
        func float(_ d: [String: Any]?, _ k: String, _ def: Float) -> Float {
            (d?[k] as? NSNumber)?.floatValue ?? def
        }
        func double(_ d: [String: Any]?, _ k: String, _ def: Double) -> Double {
            (d?[k] as? NSNumber)?.doubleValue ?? def
        }
        func bool(_ d: [String: Any]?, _ k: String, _ def: Bool) -> Bool {
            (d?[k] as? NSNumber)?.boolValue ?? def
        }

        let pre = config["preprocessor"] as? [String: Any]
        let enc = config["encoder"] as? [String: Any]
        let dec = config["decoder"] as? [String: Any]
        let pred = dec?["prednet"] as? [String: Any]
        let joint = config["joint"] as? [String: Any]
        let jn = joint?["jointnet"] as? [String: Any]
        let prompt = config["prompt"] as? [String: Any]
        let attCtx = (enc?["att_context_size"] as? [[Int]])?.last ?? [56, 13]

        guard let vocab = config["vocabulary"] as? [String] else {
            throw NemotronMlxError.invalidWeights("Missing vocabulary")
        }
        let dict = (prompt?["prompt_dictionary"] as? [String: Any])?
            .mapValues { ($0 as? NSNumber)?.intValue ?? 0 } ?? [:]

        return NemotronMlxSpec(
            sampleRate: int(pre, "sample_rate", 16_000),
            features: int(pre, "features", 128),
            nFFT: int(pre, "n_fft", 512),
            windowSize: double(pre, "window_size", 0.025),
            windowStride: double(pre, "window_stride", 0.01),
            preemph: float(pre, "preemph", 0.97),
            logGuard: float(pre, "log_zero_guard_value", 5.960464477539063e-8),
            featIn: int(enc, "feat_in", 128),
            dModel: int(enc, "d_model", 1024),
            nLayers: int(enc, "n_layers", 24),
            nHeads: int(enc, "n_heads", 8),
            ffExpansionFactor: int(enc, "ff_expansion_factor", 4),
            subsamplingFactor: int(enc, "subsampling_factor", 8),
            subsamplingConvChannels: int(enc, "subsampling_conv_channels", 256),
            convKernelSize: int(enc, "conv_kernel_size", 9),
            posEmbMaxLen: int(enc, "pos_emb_max_len", 5000),
            useBias: bool(enc, "use_bias", false),
            numPrompts: int(prompt, "num_prompts", 128),
            promptHidden: int(prompt, "prompt_hidden", 2048),
            promptDictionary: dict,
            defaultLanguage: (config["default_language"] as? String) ?? "auto",
            decoderVocabSize: int(dec, "vocab_size", 13087),
            blankAsPad: bool(dec, "blank_as_pad", true),
            predHidden: int(pred, "pred_hidden", 640),
            predRNNLayers: int(pred, "pred_rnn_layers", 2),
            jointHidden: int(jn, "joint_hidden", 640),
            jointActivation: (jn?["activation"] as? String) ?? "relu",
            encoderHidden: int(jn, "encoder_hidden", 1024),
            maxSymbols: int(config, "max_symbols", 10),
            attContextSize: attCtx,
            vocabulary: vocab
        )
    }

    // MARK: - Transcribe (cache-aware streaming encode + RNN-T greedy)

    func transcribe(audioSamples: [Float], language: String? = nil) async throws -> String {
        guard let model = self.model, let spec = self.spec else {
            throw NemotronMlxError.notLoaded
        }

        let mel = NemotronMlxMelSpectrogram(spec: spec).computeMLX(audio: audioSamples)
        guard let mel, mel.shape[1] > 0 else { throw NemotronMlxError.invalidAudio }

        // Cache-aware encode of the full utterance (native chunk size).
        let state = NemotronMlxStreamingState(encoder: model.encoder, spec: spec)
        var chunks: [MLXArray] = state.push(mel, final: true)
        if chunks.isEmpty { return "" }

        var features = concatenated(chunks, axis: 1) // [1, Ttot, dModel]
        features = model.applyPrompt(features, language: language)
        eval(features)
        let total = features.shape[1]

        // RNN-T greedy decode
        let blankID = spec.blankID
        var tokens: [Int] = []
        var lastToken: Int? = nil
        var hidden: [MLXArray?] = []
        var cell: [MLXArray?] = []
        var time = 0
        var newSymbols = 0

        while time < total {
            let feature = features[0..., time..<(time + 1), 0...]
            let (decOut, h, c) = model.decoder(y: lastToken, hidden: hidden, cell: cell)
            let jointOut = model.joint(feature, decOut) // [1,1,1,V+1]
            let logits = jointOut[0, 0, 0, 0...]
            let predToken = Int(argMax(logits).item(Int32.self))

            if predToken != blankID {
                lastToken = predToken
                hidden = h.map { Optional($0) }
                cell = c.map { Optional($0) }
                tokens.append(predToken)
                newSymbols += 1
                if spec.maxSymbols > 0, newSymbols >= spec.maxSymbols {
                    time += 1
                    newSymbols = 0
                }
            } else {
                time += 1
                newSymbols = 0
            }
        }

        return spec.decode(tokens: tokens)
    }
}
