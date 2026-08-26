import Foundation
#if arch(arm64)
import MLX

// MARK: - Qwen3-ASR MLX Engine Provider
//
// Self-contained MLX (Apple GPU) Qwen3-ASR engine inside MlxVoice.
// The app manages download / load / unload exactly like the Whisper provider.
// Model + quantization variant come from MlxSttCatalog; the matching
// mlx-community repository is resolved automatically.

final class Qwen3AsrProvider: TranscriptionProvider {
    let name = "MLX STT (Qwen3 / Parakeet)"

    var isAvailable: Bool {
        CPUArchitecture.isAppleSilicon
    }

    private(set) var isReady: Bool = false
    private var qwen3Engine: Qwen3MlxEngine?
    private var parakeetEngine: ParakeetMlxEngine?
    private var nemotronEngine: NemotronMlxEngine?
    private var glmEngine: GlmMlxEngine?
    private var funEngine: FunMlxEngine?
    private var whisperEngine: WhisperMlxEngine?
    private var loadedFamilyID: String?
    private var loadedVariantID: String?

    var modelOverride: SettingsStore.SpeechModel?

    deinit {
        // Fires before the engine properties are released, so this pass mostly
        // clears the cache as a best effort; the ASRService follows up with a
        // deferred clearCache() after the deallocation completes.
        MLX.Memory.clearCache()
    }

    init(modelOverride: SettingsStore.SpeechModel? = nil) {
        self.modelOverride = modelOverride
        // MLX keeps a growing buffer cache for reused Metal allocations. Cap it
        // so the app's footprint stays bounded between cards/transcriptions
        // (peak activations for the largest card are well below 2 GB).
        let cached = MLX.Memory.cacheMemory
        if cached > 2 * 1024 * 1024 * 1024 {
            MLX.Memory.cacheLimit = 2 * 1024 * 1024 * 1024
        }
    }

    /// Family/architecture of the selected MLX card, e.g. "qwen3-asr".
    static let mlxModelID = "qwen3-asr"

    var selectedFamilyID: String {
        SettingsStore.shared.selectedMlxSttFamilyID
    }

    var selectedVariantID: String {
        SettingsStore.shared.selectedMlxSttVariantID
    }

    func prepare(progressHandler: ((ModelPreparationProgress) -> Void)? = nil) async throws {
        try Task.checkCancellation()
        let variantID = self.selectedVariantID

        if self.isReady, self.loadedVariantID == variantID {
            return
        }

        self.isReady = false
        self.qwen3Engine = nil
        self.parakeetEngine = nil
        self.nemotronEngine = nil
        self.glmEngine = nil
        self.funEngine = nil

        let familyID = self.selectedFamilyID
        let directory = Qwen3MlxEngine.cacheDirectory(
            modelID: familyID, variantID: variantID)

        let exists: Bool
        if familyID == MlxSttFamilyKind.parakeetTdt.rawValue {
            exists = ParakeetMlxEngine.modelsExist(modelID: familyID, variantID: variantID)
        } else if familyID == MlxSttFamilyKind.nemotronAsr.rawValue {
            exists = NemotronMlxEngine.modelsExist(modelID: familyID, variantID: variantID)
        } else if familyID == MlxSttFamilyKind.glmAsr.rawValue {
            exists = GlmMlxEngine.modelsExist(modelID: familyID, variantID: variantID)
        } else if familyID == MlxSttFamilyKind.funAsr.rawValue {
            exists = FunMlxEngine.modelsExist(modelID: familyID, variantID: variantID)
        } else if familyID == MlxSttFamilyKind.whisperMlx.rawValue {
            exists = WhisperMlxEngine.modelsExist(modelID: familyID, variantID: variantID)
        } else {
            exists = Qwen3MlxEngine.modelsExist(modelID: familyID, variantID: variantID)
        }

        if !exists {
            DebugLogger.shared.info(
                "Qwen3AsrProvider: Downloading MLX card \(familyID)/\(variantID)...",
                source: "Qwen3AsrProvider"
            )
            progressHandler?(.preparingDownload)
            let progressRelay = ModelPreparationProgressRelay(progressHandler)
            if familyID == MlxSttFamilyKind.parakeetTdt.rawValue {
                try await ParakeetMlxEngine.download(
                    modelID: familyID, variantID: variantID
                ) { fraction in
                    progressRelay.report(.downloading(fraction))
                }
            } else if familyID == MlxSttFamilyKind.nemotronAsr.rawValue {
                try await NemotronMlxEngine.download(
                    modelID: familyID, variantID: variantID
                ) { fraction in
                    progressRelay.report(.downloading(fraction))
                }
            } else if familyID == MlxSttFamilyKind.glmAsr.rawValue {
                try await GlmMlxEngine.download(
                    modelID: familyID, variantID: variantID
                ) { fraction in
                    progressRelay.report(.downloading(fraction))
                }
            } else if familyID == MlxSttFamilyKind.funAsr.rawValue {
                try await FunMlxEngine.download(
                    modelID: familyID, variantID: variantID
                ) { fraction in
                    progressRelay.report(.downloading(fraction))
                }
            } else if familyID == MlxSttFamilyKind.whisperMlx.rawValue {
                try await WhisperMlxEngine.download(
                    modelID: familyID, variantID: variantID
                ) { fraction in
                    progressRelay.report(.downloading(fraction))
                }
            } else {
                try await Qwen3MlxEngine.download(
                    modelID: familyID, variantID: variantID
                ) { fraction in
                    progressRelay.report(.downloading(fraction))
                }
            }
        }

        try Task.checkCancellation()
        // Drop the previous card's cached Metal buffers before loading a new
        // engine; MLX otherwise keeps them (and their memory) around.
        MLX.Memory.clearCache()
        progressHandler?(.loading)
        DebugLogger.shared.info(
            "Qwen3AsrProvider: Loading MLX card \(familyID)/\(variantID)... (active \(Qwen3AsrProvider.bytesMB(MLX.Memory.activeMemory)) MB, cached \(Qwen3AsrProvider.bytesMB(MLX.Memory.cacheMemory)) MB)",
            source: "Qwen3AsrProvider"
        )
        if familyID == MlxSttFamilyKind.parakeetTdt.rawValue {
            let newEngine = ParakeetMlxEngine()
            try await newEngine.loadModels(
                modelID: familyID, variantID: variantID, directory: directory)
            self.parakeetEngine = newEngine
            self.qwen3Engine = nil
            self.nemotronEngine = nil
        } else if familyID == MlxSttFamilyKind.nemotronAsr.rawValue {
            let newEngine = NemotronMlxEngine()
            try await newEngine.loadModels(
                modelID: familyID, variantID: variantID, directory: directory)
            self.nemotronEngine = newEngine
            self.qwen3Engine = nil
            self.parakeetEngine = nil
            self.glmEngine = nil
        } else if familyID == MlxSttFamilyKind.glmAsr.rawValue {
            let newEngine = GlmMlxEngine()
            try await newEngine.loadModels(
                modelID: familyID, variantID: variantID, directory: directory)
            self.glmEngine = newEngine
            self.qwen3Engine = nil
            self.parakeetEngine = nil
            self.nemotronEngine = nil
            self.funEngine = nil
        } else if familyID == MlxSttFamilyKind.funAsr.rawValue {
            let newEngine = FunMlxEngine()
            try await newEngine.loadModels(
                modelID: familyID, variantID: variantID, directory: directory)
            self.funEngine = newEngine
            self.qwen3Engine = nil
            self.parakeetEngine = nil
            self.nemotronEngine = nil
            self.glmEngine = nil
        } else if familyID == MlxSttFamilyKind.whisperMlx.rawValue {
            let newEngine = WhisperMlxEngine()
            try await newEngine.loadModels(
                modelID: familyID, variantID: variantID, directory: directory)
            self.whisperEngine = newEngine
            self.qwen3Engine = nil
            self.parakeetEngine = nil
            self.nemotronEngine = nil
            self.glmEngine = nil
            self.funEngine = nil
        } else {
            let newEngine = Qwen3MlxEngine()
            try await newEngine.loadModels(
                modelID: familyID, variantID: variantID, directory: directory)
            self.qwen3Engine = newEngine
            self.parakeetEngine = nil
            self.nemotronEngine = nil
            self.glmEngine = nil
            self.funEngine = nil
        }
        self.loadedFamilyID = familyID
        self.loadedVariantID = variantID
        self.isReady = true
        DebugLogger.shared.info("Qwen3AsrProvider: MLX model loaded and ready!", source: "Qwen3AsrProvider")
    }

    func transcribe(_ samples: [Float]) async throws -> ASRTranscriptionResult {
        guard self.isAvailable else {
            throw NSError(domain: "Qwen3AsrProvider", code: -1, userInfo: [NSLocalizedDescriptionKey: "MLX STT requires Apple Silicon"])
        }
        guard !samples.isEmpty else {
            return ASRTranscriptionResult(text: "")
        }

        // Self-heal: if the selected card changed, (re)load before transcribing.
        if !self.isReady || self.loadedFamilyID != self.selectedFamilyID
            || self.loadedVariantID != self.selectedVariantID
        {
            try await self.prepare()
        }

        let familyID = self.selectedFamilyID
        if familyID == MlxSttFamilyKind.nemotronAsr.rawValue {
            guard let engine = self.nemotronEngine else {
                throw NSError(domain: "Qwen3AsrProvider", code: -2, userInfo: [NSLocalizedDescriptionKey: "Nemotron model not loaded"])
            }
            DebugLogger.shared.info("Qwen3AsrProvider: Starting Nemotron transcription for \(samples.count) samples...", source: "Qwen3AsrProvider")
            do {
                let text = try await engine.transcribe(audioSamples: samples)
                let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
                DebugLogger.shared.info("Qwen3AsrProvider: Transcribed text: '\(trimmed)'", source: "Qwen3AsrProvider")
                return ASRTranscriptionResult(text: trimmed, confidence: 1.0)
            } catch {
                DebugLogger.shared.error("Qwen3AsrProvider: Nemotron transcription failed safely with error: \(error.localizedDescription)", source: "Qwen3AsrProvider")
                throw error
            }
        }

        if familyID == MlxSttFamilyKind.glmAsr.rawValue {
            guard let engine = self.glmEngine else {
                throw NSError(domain: "Qwen3AsrProvider", code: -2, userInfo: [NSLocalizedDescriptionKey: "GLM model not loaded"])
            }
            DebugLogger.shared.info("Qwen3AsrProvider: Starting GLM transcription for \(samples.count) samples...", source: "Qwen3AsrProvider")
            do {
                let text = try await engine.transcribe(audioSamples: samples)
                let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
                DebugLogger.shared.info("Qwen3AsrProvider: Transcribed text: '\(trimmed)'", source: "Qwen3AsrProvider")
                return ASRTranscriptionResult(text: trimmed, confidence: 1.0)
            } catch {
                DebugLogger.shared.error("Qwen3AsrProvider: GLM transcription failed safely with error: \(error.localizedDescription)", source: "Qwen3AsrProvider")
                throw error
            }
        }

        if familyID == MlxSttFamilyKind.whisperMlx.rawValue {
            guard let engine = self.whisperEngine else {
                throw NSError(domain: "Qwen3AsrProvider", code: -2, userInfo: [NSLocalizedDescriptionKey: "Whisper model not loaded"])
            }
            DebugLogger.shared.info("Qwen3AsrProvider: Starting Whisper transcription for \(samples.count) samples...", source: "Qwen3AsrProvider")
            do {
                let text = try await engine.transcribe(audioSamples: samples)
                let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
                DebugLogger.shared.info("Qwen3AsrProvider: Transcribed text: '\(trimmed)'", source: "Qwen3AsrProvider")
                return ASRTranscriptionResult(text: trimmed, confidence: 1.0)
            } catch {
                DebugLogger.shared.error("Qwen3AsrProvider: Whisper transcription failed safely with error: \(error.localizedDescription)", source: "Qwen3AsrProvider")
                throw error
            }
        }

        if familyID == MlxSttFamilyKind.funAsr.rawValue {
            guard let engine = self.funEngine else {
                throw NSError(domain: "Qwen3AsrProvider", code: -2, userInfo: [NSLocalizedDescriptionKey: "Fun-ASR model not loaded"])
            }
            DebugLogger.shared.info("Qwen3AsrProvider: Starting Fun-ASR transcription for \(samples.count) samples...", source: "Qwen3AsrProvider")
            do {
                let text = try await engine.transcribe(audioSamples: samples)
                let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
                DebugLogger.shared.info("Qwen3AsrProvider: Transcribed text: '\(trimmed)'", source: "Qwen3AsrProvider")
                return ASRTranscriptionResult(text: trimmed, confidence: 1.0)
            } catch {
                DebugLogger.shared.error("Qwen3AsrProvider: Fun-ASR transcription failed safely with error: \(error.localizedDescription)", source: "Qwen3AsrProvider")
                throw error
            }
        }

        let isParakeet = familyID == MlxSttFamilyKind.parakeetTdt.rawValue
        if isParakeet {
            guard let engine = self.parakeetEngine else {
                throw NSError(domain: "Qwen3AsrProvider", code: -2, userInfo: [NSLocalizedDescriptionKey: "Parakeet model not loaded"])
            }
            DebugLogger.shared.info("Qwen3AsrProvider: Starting Parakeet transcription for \(samples.count) samples...", source: "Qwen3AsrProvider")
            do {
                let text = try await engine.transcribe(audioSamples: samples)
                let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
                DebugLogger.shared.info("Qwen3AsrProvider: Transcribed text: '\(trimmed)'", source: "Qwen3AsrProvider")
                return ASRTranscriptionResult(text: trimmed, confidence: 1.0)
            } catch {
                DebugLogger.shared.error("Qwen3AsrProvider: Parakeet transcription failed safely with error: \(error.localizedDescription)", source: "Qwen3AsrProvider")
                throw error
            }
        }

        guard let engine = self.qwen3Engine else {
            throw NSError(domain: "Qwen3AsrProvider", code: -2, userInfo: [NSLocalizedDescriptionKey: "Qwen3-ASR model not loaded"])
        }
        let maxTokens = min(SettingsStore.shared.asrContextTokenLimit, 256)
        let contextWords = SettingsStore.shared.qwen3ContextWords
            .trimmingCharacters(in: .whitespacesAndNewlines)
        DebugLogger.shared.info("Qwen3AsrProvider: Starting MLX transcription for \(samples.count) samples (maxTokens: \(maxTokens))...", source: "Qwen3AsrProvider")
        do {
            let text = try await engine.transcribe(
                audioSamples: samples,
                language: nil,
                maxNewTokens: maxTokens,
                contextWords: contextWords.isEmpty ? nil : contextWords
            )
            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            DebugLogger.shared.info("Qwen3AsrProvider: Transcribed text: '\(trimmed)' (MLX active \(Qwen3AsrProvider.bytesMB(MLX.Memory.activeMemory)) MB, cached \(Qwen3AsrProvider.bytesMB(MLX.Memory.cacheMemory)) MB, peak \(Qwen3AsrProvider.bytesMB(MLX.Memory.peakMemory)) MB)", source: "Qwen3AsrProvider")
            return ASRTranscriptionResult(text: trimmed, confidence: 1.0)
        } catch {
            DebugLogger.shared.error("Qwen3AsrProvider: Transcription failed safely with error: \(error.localizedDescription)", source: "Qwen3AsrProvider")
            throw error
        }
    }

    func transcribeFinal(_ samples: [Float]) async throws -> ASRTranscriptionResult {
        try await self.transcribe(samples)
    }

    func clearCache() async throws {
        self.isReady = false
        self.qwen3Engine = nil
        self.parakeetEngine = nil
        self.nemotronEngine = nil
        self.glmEngine = nil
        self.funEngine = nil
        self.loadedFamilyID = nil
        self.loadedVariantID = nil
        MLX.Memory.clearCache()

        let root = Qwen3MlxEngine.modelsRootDirectory()
            .appendingPathComponent("MlxStt", isDirectory: true)
        if FileManager.default.fileExists(atPath: root.path) {
            try? FileManager.default.removeItem(at: root)
            DebugLogger.shared.info("Qwen3AsrProvider: Deleted MLX model cache", source: "Qwen3AsrProvider")
        }
    }

    private static func bytesMB(_ bytes: Int) -> String {
        String(format: "%.0f", Double(bytes) / 1_048_576)
    }

    func modelsExistOnDisk() -> Bool {
        let familyID = self.selectedFamilyID
        if familyID == MlxSttFamilyKind.parakeetTdt.rawValue {
            return ParakeetMlxEngine.modelsExist(
                modelID: familyID, variantID: self.selectedVariantID)
        }
        if familyID == MlxSttFamilyKind.nemotronAsr.rawValue {
            return NemotronMlxEngine.modelsExist(
                modelID: familyID, variantID: self.selectedVariantID)
        }
        if familyID == MlxSttFamilyKind.glmAsr.rawValue {
            return GlmMlxEngine.modelsExist(
                modelID: familyID, variantID: self.selectedVariantID)
        }
        if familyID == MlxSttFamilyKind.funAsr.rawValue {
            return FunMlxEngine.modelsExist(
                modelID: familyID, variantID: self.selectedVariantID)
        }
        return Qwen3MlxEngine.modelsExist(
            modelID: familyID, variantID: self.selectedVariantID)
    }
}
#else
final class Qwen3AsrProvider: TranscriptionProvider {
    let name = "Qwen3 ASR (Unavailable on Intel)"
    var isAvailable: Bool { false }
    var isReady: Bool { false }
    var modelOverride: SettingsStore.SpeechModel?

    init(modelOverride: SettingsStore.SpeechModel? = nil) {
        self.modelOverride = modelOverride
    }

    func prepare(progressHandler: ((ModelPreparationProgress) -> Void)?) async throws {
        throw NSError(domain: "Qwen3AsrProvider", code: -1, userInfo: [NSLocalizedDescriptionKey: "Qwen3-ASR requires Apple Silicon"])
    }
    func transcribe(_ samples: [Float]) async throws -> ASRTranscriptionResult {
        throw NSError(domain: "Qwen3AsrProvider", code: -1, userInfo: [NSLocalizedDescriptionKey: "Qwen3-ASR requires Apple Silicon"])
    }
    func transcribeFinal(_ samples: [Float]) async throws -> ASRTranscriptionResult {
        throw NSError(domain: "Qwen3AsrProvider", code: -1, userInfo: [NSLocalizedDescriptionKey: "Qwen3-ASR requires Apple Silicon"])
    }
    func clearCache() async throws {}
    func modelsExistOnDisk() -> Bool { false }
}
#endif
