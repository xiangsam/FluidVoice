import AppKit
import Combine
import SwiftUI

@MainActor
final class VoiceEngineSettingsViewModel: ObservableObject {
    let settings: SettingsStore
    private let appServices: AppServices
    private var cancellables = Set<AnyCancellable>()

    var asr: ASRService { self.appServices.asr }

    var areSpeechModelActionsBlocked: Bool {
        self.asr.isRunning
            || self.downloadingModel != nil
            || self.asr.hasActiveModelDownload
            || self.asr.hasActiveModelPreparation
            || self.asr.isCancellingModelPreparation
            || (!self.asr.isAsrReady && (self.asr.isDownloadingModel || self.asr.isLoadingModel))
    }

    @Published var modelSortOption: ModelSortOption = .provider
    @Published var providerFilter: SpeechProviderFilter = .all
    @Published var englishOnlyFilter: Bool = false
    @Published var installedOnlyFilter: Bool = false
    @Published var showSpeechFilters: Bool = false

    @Published var selectedSpeechProvider: SettingsStore.SpeechModel.Provider
    @Published var previewSpeechModel: SettingsStore.SpeechModel
    @Published var isConfigPanelExpanded: Bool = false
    @Published var showAdvancedSpeechInfo: Bool = false
    @Published var suppressSpeechProviderSync: Bool = false
    @Published var skipNextSpeechModelSync: Bool = false

    var downloadingModel: SettingsStore.SpeechModel? {
        guard let modelID = self.asr.downloadingModelId else { return nil }
        return SettingsStore.SpeechModel.allCases.first { $0.id == modelID }
    }

    var downloadProgress: Double {
        self.asr.downloadProgress ?? 0.0
    }

    /// True while a specific MLX card is being downloaded (the card must be
    /// the selected one, since downloads run through the selected model).
    func isMlxCardDownloading(_ card: MlxSttCard) -> Bool {
        guard self.asr.downloadingModelId == SettingsStore.SpeechModel.qwen3Asr.id else {
            return false
        }
        return self.settings.selectedMlxSttCardID == card.pathID
    }

    var isCancellingModelDownload: Bool {
        self.asr.isCancellingModelDownload
    }

    @Published var removeFillerWordsEnabled: Bool

    init(settings: SettingsStore, appServices: AppServices) {
        self.settings = settings
        self.appServices = appServices
        self.previewSpeechModel = settings.selectedSpeechModel
        self.selectedSpeechProvider = settings.selectedSpeechModel.provider
        self.removeFillerWordsEnabled = settings.removeFillerWordsEnabled
        appServices.objectWillChange
            .sink { [weak self] _ in
                Task { @MainActor in
                    self?.objectWillChange.send()
                }
            }
            .store(in: &self.cancellables)
    }

    func onAppear() {
        self.previewSpeechModel = self.settings.selectedSpeechModel
        self.selectedSpeechProvider = self.settings.selectedSpeechModel.provider
        self.removeFillerWordsEnabled = self.settings.removeFillerWordsEnabled
        self.isConfigPanelExpanded = false

        Task {
            await self.asr.checkIfModelsExistAsync()
        }
    }

    func handleSelectedSpeechModelChange(_ newValue: SettingsStore.SpeechModel) {
        if self.skipNextSpeechModelSync {
            self.skipNextSpeechModelSync = false
            return
        }
        guard !self.suppressSpeechProviderSync else { return }
        self.previewSpeechModel = newValue
        self.setSelectedSpeechProvider(newValue.provider)
    }

    var filteredSpeechModels: [SettingsStore.SpeechModel] {
        var models = SettingsStore.SpeechModel.availableModels

        switch self.providerFilter {
        case .all:
            break
        case .cloud:
            models = models.filter { $0.provider == .cloud }
        case .apple:
            models = models.filter { $0.provider == .apple }
        case .openai:
            models = models.filter { $0.provider == .openai }
        case .qwen:
            models = models.filter { $0.provider == .qwen }
        }

        if self.englishOnlyFilter {
            models = models.filter { model in
                let label = model.languageSupport.lowercased()
                let title = model.humanReadableName.lowercased()
                return label.contains("english only") || title.contains("english")
            }
        }

        if self.installedOnlyFilter {
            models = models.filter { $0.isInstalled }
        }

        switch self.modelSortOption {
        case .provider:
            models.sort { $0.brandName.localizedCaseInsensitiveCompare($1.brandName) == .orderedAscending }
        case .accuracy:
            models.sort { $0.accuracyPercent > $1.accuracyPercent }
        case .speed:
            models.sort { $0.speedPercent > $1.speedPercent }
        }

        return models
    }

    func activateSpeechModel(_ model: SettingsStore.SpeechModel) {
        guard !self.areSpeechModelActionsBlocked else { return }
        withAnimation(.spring(response: 0.3, dampingFraction: 0.7)) {
            self.settings.selectedSpeechModel = model
            self.previewSpeechModel = model
            self.setSelectedSpeechProvider(model.provider)
            // Exclusive selection: only ONE engine can be active at a time.
            // When the user picks Apple or a cloud model, clear the MLX card
            // ID so the local card list shows no card as selected. The MLX
            // card ID is re-pointed when an MLX card is picked again.
            if !model.isMlxEngineModel {
                self.settings.selectedMlxSttCardID = ""
            }
        }
        // Exclusive engine selection: the ASR routing is driven by
        // selectedSpeechModel and (for MLX cards) selectedMlxSttCardID only.
        // Whenever the user picks a different model, force a provider reset so
        // a stale cloud/Apple provider can never keep serving transcription
        // after the switch back to a local card.
        self.asr.resetTranscriptionProvider()
        Task {
            do {
                try await self.asr.ensureAsrReady(source: .settings)
            } catch is CancellationError {
                DebugLogger.shared.info("Model activation cancelled: \(model.displayName)", source: "AISettingsView")
            } catch {
                DebugLogger.shared.error("Failed to prepare model after activation: \(error)", source: "AISettingsView")
                self.asr.errorTitle = "Model Activation Failed"
                self.asr.errorMessage = error.localizedDescription
                self.asr.showError = true
            }
        }
    }

    // MARK: - MLX STT Cards (per-card lifecycle)

    /// Activate an MLX card: select it as the active MLX model and point the
    /// engine at the card's repository. Only blocked while a download /
    /// preparation is in flight; otherwise the click always applies and takes
    /// effect on the next transcription via the provider's self-heal.
    func selectMlxCard(_ card: MlxSttCard) {
        // A background download must not lock model switching: the user can
        // keep using any installed card while another downloads. Only block
        // selection while the target model itself is being prepared/cancelled.
        guard !self.asr.hasActiveModelPreparation,
            !self.asr.isCancellingModelPreparation
        else { return }
        // Switching to a local MLX card is exclusive: point the MLX card ID at
        // the new card AND switch the unified SpeechModel to the MLX engine so
        // the ASR routing can never keep serving a cloud/Apple engine.
        self.settings.selectedMlxSttCardID = card.pathID
        self.activateSpeechModel(.qwen3Asr)
    }

    /// Download a specific MLX card (selects it first, then downloads).
    /// If the download fails, the previous card selection is restored so the
    /// UI never reports a card as "in use" while it is not installed.
    func downloadMlxCard(_ card: MlxSttCard) {
        guard !self.areSpeechModelActionsBlocked else { return }
        let previousCardID = self.settings.selectedMlxSttCardID
        self.settings.selectedMlxSttCardID = card.pathID
        self.settings.selectedSpeechModel = .qwen3Asr
        self.asr.resetTranscriptionProvider()
        self.downloadSpeechModel(.qwen3Asr) { [weak self] in
            guard let self else { return }
            // Only restore when the user hasn't already switched somewhere else.
            if self.settings.selectedMlxSttCardID == card.pathID {
                self.settings.selectedMlxSttCardID = previousCardID
            }
        }
    }

    /// Uninstall one MLX card from disk.
    func uninstallMlxCard(_ card: MlxSttCard) {
        guard !self.areSpeechModelActionsBlocked else { return }
        Task { [weak self] in
            await self?.asr.deleteMlxCard(pathID: card.pathID)
        }
    }

    func downloadSpeechModel(
        _ model: SettingsStore.SpeechModel,
        onFailure: (() -> Void)? = nil
    ) {
        guard !self.areSpeechModelActionsBlocked else { return }
        Task { [weak self] in
            guard let self else { return }
            do {
                try await self.asr.downloadModel(model, progressHandler: nil)
                DebugLogger.shared.info("Model download completed: \(model.displayName)", source: "VoiceEngineVM")
            } catch is CancellationError {
                DebugLogger.shared.info("Model download cancelled: \(model.displayName)", source: "VoiceEngineVM")
                onFailure?()
            } catch {
                DebugLogger.shared.error("Failed to download model \(model.displayName): \(error)", source: "VoiceEngineVM")
                self.asr.errorTitle = "Model Download Failed"
                self.asr.errorMessage = error.localizedDescription
                self.asr.showError = true
                onFailure?()
            }
        }
    }

    func cancelSpeechModelDownload() {
        guard self.downloadingModel != nil, !self.isCancellingModelDownload else { return }
        self.asr.cancelModelDownload()
    }

    func cancelActiveModelPreparation() {
        self.asr.cancelModelPreparation()
    }

    func deleteSpeechModel(_ model: SettingsStore.SpeechModel) {
        guard !self.areSpeechModelActionsBlocked else { return }
        let previousActive = self.settings.selectedSpeechModel

        Task {
            let shouldRestore = previousActive != model
            await MainActor.run {
                if shouldRestore {
                    self.suppressSpeechProviderSync = true
                }
                self.settings.selectedSpeechModel = model
                self.asr.resetTranscriptionProvider()
            }

            defer {
                Task { @MainActor in
                    guard shouldRestore else { return }
                    self.skipNextSpeechModelSync = true
                    self.settings.selectedSpeechModel = previousActive
                    self.asr.resetTranscriptionProvider()
                    if self.previewSpeechModel == model {
                        self.previewSpeechModel = model
                    }
                    self.suppressSpeechProviderSync = false
                }
            }

            await self.deleteModels()
        }
    }

    func isActiveSpeechModel(_ model: SettingsStore.SpeechModel) -> Bool {
        self.settings.selectedSpeechModel == model
    }

    var modelDescriptionText: String {
        let model = self.settings.selectedSpeechModel
        switch model {
        case .appleSpeech:
            return "Apple Speech (Legacy) uses built-in macOS speech recognition. No model download required, works on Intel and Apple Silicon."
        case .appleSpeechAnalyzer:
            return "Apple Speech uses advanced on-device recognition with fast, accurate transcription. Requires macOS 26+."
        case .qwen3Asr:
            return "Qwen3 ASR is a multilingual local model with strong quality, but higher memory usage. Requires macOS 15+."
        default:
            return "Whisper models support 99 languages and work on any Mac."
        }
    }

    func downloadModels() async {
        do {
            try await self.asr.ensureAsrReady(source: .settings)
        } catch is CancellationError {
            DebugLogger.shared.info("Model download cancelled", source: "AISettingsView")
        } catch {
            DebugLogger.shared.error("Failed to download models: \(error)", source: "AISettingsView")
            self.asr.errorTitle = "Model Download Failed"
            self.asr.errorMessage = error.localizedDescription
            self.asr.showError = true
        }
    }

    func deleteModels() async {
        do {
            try await self.asr.clearModelCache()
            await self.asr.checkIfModelsExistAsync()
        } catch {
            DebugLogger.shared.error("Failed to delete models: \(error)", source: "AISettingsView")
        }
    }

    func deleteModel(_ model: SettingsStore.SpeechModel) async {
        do {
            try await self.asr.clearModelCache(for: model)
            await self.asr.checkIfModelsExistAsync()
        } catch {
            DebugLogger.shared.error("Failed to delete model \(model.displayName): \(error)", source: "VoiceEngineSettingsViewModel")
        }
    }

    func setSelectedSpeechProvider(_ provider: SettingsStore.SpeechModel.Provider) {
        self.selectedSpeechProvider = provider
    }

}
