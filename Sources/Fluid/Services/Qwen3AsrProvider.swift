import Foundation
#if arch(arm64) && canImport(FluidAudio)
import FluidAudio

/// TranscriptionProvider implementation for Qwen3-ASR (0.6B) running via FluidAudio on Apple Silicon.
final class Qwen3AsrProvider: TranscriptionProvider {
    let name = "Qwen3 ASR (0.6B CoreML)"

    var isAvailable: Bool {
        CPUArchitecture.isAppleSilicon
    }

    private(set) var isReady: Bool = false
    private var manager: Qwen3AsrManager?
    private var loadedVariant: Qwen3AsrVariant?

    var modelOverride: SettingsStore.SpeechModel?

    init(modelOverride: SettingsStore.SpeechModel? = nil) {
        self.modelOverride = modelOverride
    }

    private var currentVariant: Qwen3AsrVariant {
        SettingsStore.shared.qwen3AsrVariant == .int8 ? .int8 : .f32
    }

    func prepare(progressHandler: ((ModelPreparationProgress) -> Void)? = nil) async throws {
        try Task.checkCancellation()
        let variant = self.currentVariant

        if self.isReady, self.loadedVariant == variant, self.manager != nil {
            return
        }

        self.isReady = false
        self.manager = nil

        let targetDir = Qwen3AsrModels.defaultCacheDirectory(variant: variant)
        if !Qwen3AsrModels.modelsExist(at: targetDir) {
            DebugLogger.shared.info("Qwen3AsrProvider: Downloading Qwen3 ASR (\(variant.rawValue))...", source: "Qwen3AsrProvider")
            progressHandler?(.preparingDownload)

            let progressRelay = ModelPreparationProgressRelay(progressHandler)
            let fluidProgressHandler: DownloadUtils.ProgressHandler = { progress in
                switch progress.phase {
                case .listing:
                    progressRelay.report(.preparingDownload)
                case .downloading:
                    progressRelay.report(.downloading(progress.fractionCompleted))
                case .compiling:
                    progressRelay.report(.optimizing)
                }
            }

            try await Qwen3AsrModels.download(
                variant: variant,
                to: targetDir,
                force: false,
                progressHandler: fluidProgressHandler
            )
        }

        try Task.checkCancellation()
        progressHandler?(.loading)

        DebugLogger.shared.info("Qwen3AsrProvider: Loading Qwen3 ASR models from \(targetDir.path)...", source: "Qwen3AsrProvider")
        let asrManager = Qwen3AsrManager()
        try await asrManager.loadModels(from: targetDir)

        self.manager = asrManager
        self.loadedVariant = variant
        self.isReady = true
        DebugLogger.shared.info("Qwen3AsrProvider: Model loaded and ready!", source: "Qwen3AsrProvider")
    }

    func transcribe(_ samples: [Float]) async throws -> ASRTranscriptionResult {
        guard self.isAvailable else {
            throw NSError(domain: "Qwen3AsrProvider", code: -1, userInfo: [NSLocalizedDescriptionKey: "Qwen3 ASR requires Apple Silicon"])
        }
        guard let manager = self.manager else {
            throw NSError(domain: "Qwen3AsrProvider", code: -2, userInfo: [NSLocalizedDescriptionKey: "Qwen3 ASR model not loaded"])
        }
        guard !samples.isEmpty else {
            return ASRTranscriptionResult(text: "")
        }

        let optLang: String? = nil
        DebugLogger.shared.info("Qwen3AsrProvider: Starting transcription for \(samples.count) samples...", source: "Qwen3AsrProvider")
        let text = try await manager.transcribe(audioSamples: samples, language: optLang, maxNewTokens: 256)
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        DebugLogger.shared.info("Qwen3AsrProvider: Transcribed text: '\(trimmed)'", source: "Qwen3AsrProvider")
        return ASRTranscriptionResult(text: trimmed, confidence: 1.0)
    }

    func transcribeFinal(_ samples: [Float]) async throws -> ASRTranscriptionResult {
        try await self.transcribe(samples)
    }

    func clearCache() async throws {
        self.isReady = false
        self.manager = nil
        self.loadedVariant = nil

        let int8Dir = Qwen3AsrModels.defaultCacheDirectory(variant: .int8)
        let f32Dir = Qwen3AsrModels.defaultCacheDirectory(variant: .f32)

        if FileManager.default.fileExists(atPath: int8Dir.path) {
            try? FileManager.default.removeItem(at: int8Dir)
            DebugLogger.shared.info("Qwen3AsrProvider: Deleted Qwen3 int8 model cache", source: "Qwen3AsrProvider")
        }
        if FileManager.default.fileExists(atPath: f32Dir.path) {
            try? FileManager.default.removeItem(at: f32Dir)
            DebugLogger.shared.info("Qwen3AsrProvider: Deleted Qwen3 f32 model cache", source: "Qwen3AsrProvider")
        }
    }

    func modelsExistOnDisk() -> Bool {
        let targetDir = Qwen3AsrModels.defaultCacheDirectory(variant: self.currentVariant)
        return Qwen3AsrModels.modelsExist(at: targetDir)
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
        throw NSError(domain: "Qwen3AsrProvider", code: -1, userInfo: [NSLocalizedDescriptionKey: "Qwen3 ASR requires Apple Silicon"])
    }
    func transcribe(_ samples: [Float]) async throws -> ASRTranscriptionResult {
        throw NSError(domain: "Qwen3AsrProvider", code: -1, userInfo: [NSLocalizedDescriptionKey: "Qwen3 ASR requires Apple Silicon"])
    }
    func transcribeFinal(_ samples: [Float]) async throws -> ASRTranscriptionResult {
        throw NSError(domain: "Qwen3AsrProvider", code: -1, userInfo: [NSLocalizedDescriptionKey: "Qwen3 ASR requires Apple Silicon"])
    }
    func clearCache() async throws {}
    func modelsExistOnDisk() -> Bool { false }
}
#endif
