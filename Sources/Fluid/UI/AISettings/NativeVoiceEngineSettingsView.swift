//
//  NativeVoiceEngineSettingsView.swift
//  MlxVoice
//
//  Modern, card-grouped native SwiftUI Voice Engine Settings View adhering to Apple HIG.
//

import SwiftUI

typealias SpeechModel = SettingsStore.SpeechModel

enum ModelCategoryTab: String, CaseIterable, Identifiable {
    case apple = "Apple 内置"
    case local = "本地离线"
    case cloud = "云端与局域网"

    var id: String { self.rawValue }

    var icon: String {
        switch self {
        case .apple: return "apple.logo"
        case .local: return "cpu"
        case .cloud: return "network"
        }
    }
}

struct NativeVoiceEngineSettingsView: View {
    @ObservedObject var viewModel: VoiceEngineSettingsViewModel
    @ObservedObject var settings: SettingsStore
    let theme: AppTheme

    @State private var selectedTab: ModelCategoryTab = .apple
    @State private var isTestingConnection: Bool = false
    @State private var connectionTestMessage: String? = nil
    @State private var connectionTestSuccess: Bool? = nil
    @State private var fetchedOllamaModels: [String] = []
    @State private var isFetchingOllamaModels = false
    @State private var ollamaFetchErrorMessage: String? = nil
    @State private var fetchedOpenRouterModels: [String] = []
    @State private var isFetchingOpenRouterModels = false
    @State private var openRouterFetchErrorMessage: String? = nil

    /// Curated OpenRouter transcription-capable model IDs (same set as the
    /// site filter `output_modalities=transcription`). Used as the fallback
    /// list when the feed cannot be fetched.
    private static let curatedOpenRouterSTTModels: [String] = [
        "openai/gpt-4o-mini-transcribe",
        "openai/gpt-4o-transcribe",
        "openai/gpt-transcribe",
        "openai/whisper-large-v3-turbo",
        "openai/whisper-large-v3",
        "openai/whisper-1",
        "nvidia/nemotron-3.5-asr-streaming-multilingual-0.6b",
        "nvidia/parakeet-tdt-0.6b-v3",
        "qwen/qwen3-asr-0.6b",
        "qwen/qwen3-asr-1.7b",
        "qwen/qwen3-asr-flash-2026-02-10",
        "mistralai/voxtral-mini-3b-2507",
        "mistralai/voxtral-mini-transcribe",
        "mistralai/voxtral-small-24b-2507-stt",
        "fish-audio/transcribe-1",
        "deepgram/nova-3",
        "google/chirp-3",
        "microsoft/mai-transcribe-1.5",
        "x-ai/grok-stt-1.0",
    ]

    private var activeModel: SpeechModel {
        self.settings.selectedSpeechModel
    }

    /// Move the category picker to the tab matching the active model.
    private func syncTabWithActiveModel() {
        if self.activeModel == .appleSpeech || self.activeModel == .appleSpeechAnalyzer {
            self.selectedTab = .apple
        } else if self.activeModel.isCloudModel {
            self.selectedTab = .cloud
        } else {
            self.selectedTab = .local
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            // 1. Header
            self.headerSection

            // 2. Active Model Hero Card
            self.activeHeroCard

            // 3. Category Segmented Control
            self.categoryPickerSection

            // 4. Model Cards Group
            self.modelGroupSection

            // 5. Intelligent Fallback Settings (Only for Cloud/Ollama models)
            if self.activeModel.isCloudModel {
                self.smartFallbackSection
            }

            // 6. Helpful Tip
            self.tipSection
        }
        .padding(.horizontal, 4)
        .padding(.vertical, 8)
        .onAppear {
            self.viewModel.onAppear()
            self.syncTabWithActiveModel()
        }
        .onChange(of: self.settings.selectedSpeechModel) { _, _ in
            // Tab must follow the active engine so a stale cloud tab never
            // looks "alive" after switching back to a local model.
            withAnimation(.easeInOut(duration: 0.15)) {
                self.syncTabWithActiveModel()
            }
        }
    }

    // MARK: - 1. Header
    private var headerSection: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 8) {
                Image(systemName: "waveform.circle.fill")
                    .font(.system(size: 24))
                    .foregroundStyle(self.theme.palette.accent)
                Text("Voice Engine".loc)
                    .font(.title2)
                    .fontWeight(.bold)
            }
            Text("Select and configure the speech recognition engine used to convert your voice to text.".loc)
                .font(.subheadline)
                .foregroundStyle(.secondary)
        }
    }

    // MARK: - 2. Active Model Hero Card
    private var activeHeroCard: some View {
        VStack(spacing: 12) {
            HStack(spacing: 14) {
                ZStack {
                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .fill(Color.fluidGreen.opacity(0.15))
                        .frame(width: 48, height: 48)
                    Image(systemName: self.iconForModel(self.activeModel))
                        .font(.system(size: 22))
                        .foregroundStyle(Color.fluidGreen)
                }

                VStack(alignment: .leading, spacing: 3) {
                    HStack(spacing: 6) {
                        Text(self.displayNameForModel(self.activeModel))
                            .font(.headline)
                            .fontWeight(.semibold)

                        Text("Current Active".loc)
                            .font(.system(size: 10, weight: .bold))
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(Capsule().fill(Color.fluidGreen.opacity(0.2)))
                            .foregroundStyle(Color.fluidGreen)
                    }

                    Text(self.subtitleForModel(self.activeModel))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                Spacer()
            }

            Divider().opacity(0.3)

            // Status Bar Indicator
            HStack(spacing: 6) {
                if self.activeModel.isCloudModel {
                    Image(systemName: self.settings.cloudSTTAutoFallback ? "shield.checkmark.fill" : "network")
                        .foregroundStyle(self.theme.palette.accent)
                        .font(.caption)
                    Text(self.settings.cloudSTTAutoFallback ? "智能容灾已就绪：网络异常或超时将无缝切换至 Apple Speech 兜底识别".loc : "当前运行在远程/局域网 ASR 模式".loc)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                } else if self.activeModel == .appleSpeech || self.activeModel == .appleSpeechAnalyzer {
                    Image(systemName: "apple.logo")
                        .foregroundStyle(Color.fluidGreen)
                        .font(.caption)
                    Text("macOS 系统级内置引擎：零延迟、无需下载、完全离线安全".loc)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                } else {
                    Image(systemName: "cpu.fill")
                        .foregroundStyle(Color.fluidGreen)
                        .font(.caption)
                    Text("本地神经网络引擎：完全离线运行于 Apple Silicon，保护隐私".loc)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
            }
        }
        .padding(14)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color(nsColor: .controlBackgroundColor))
                .overlay(
                    RoundedRectangle(cornerRadius: 12)
                        .stroke(Color.secondary.opacity(0.18), lineWidth: 1)
                )
        )
    }

    // MARK: - 3. Category Picker Section
    private var categoryPickerSection: some View {
        Picker("", selection: self.$selectedTab) {
            ForEach(ModelCategoryTab.allCases) { tab in
                Label(tab.rawValue.loc, systemImage: tab.icon).tag(tab)
            }
        }
        .pickerStyle(.segmented)
    }

    // MARK: - 4. Model Group Section
    @ViewBuilder
    private var modelGroupSection: some View {
        switch self.selectedTab {
        case .apple:
            VStack(alignment: .leading, spacing: 10) {
                Label("Apple Built-in Engines (Zero Download)".loc, systemImage: "apple.logo")
                    .font(.headline)
                    .foregroundStyle(.primary)

                VStack(spacing: 10) {
                    self.modelCard(for: .appleSpeechAnalyzer)
                    self.modelCard(for: .appleSpeech)
                }
            }

        case .local:
            VStack(alignment: .leading, spacing: 16) {
                // Context Length & Dynamic Memory Estimator
                self.contextMemoryEstimatorSection

                // Download Mirror Acceleration Bar
                self.downloadAccelerationSection

                // MLX Engine Group (built into MlxVoice, independent of the
                // FluidAudio package): one engine, flat per-model cards.
                self.mlxEngineGroupSection
            }

        case .cloud:
            VStack(alignment: .leading, spacing: 10) {
                Label("Cloud & LAN Servers (Ollama / Cloud APIs)".loc, systemImage: "network")
                    .font(.headline)
                    .foregroundStyle(.primary)

                VStack(spacing: 10) {
                    self.modelCard(for: .cloudOllama)
                    self.modelCard(for: .cloudOpenAI)
                    self.modelCard(for: .cloudGroq)
                    self.modelCard(for: .cloudOpenRouter)
                    self.modelCard(for: .cloudCustom)
                }
            }
        }
    }

    // MARK: - MLX Engine Group (flat per-card selection)

    private var selectedMlxCard: MlxSttCard? {
        MlxSttCatalog.card(pathID: self.settings.selectedMlxSttCardID)
    }

    /// One engine, flat model cards: Qwen3-ASR / Parakeet / Nemotron /
    /// GLM-ASR / Fun-ASR / Whisper all live under the single built-in MLX
    /// engine and are selected by clicking their card row directly.
    private var mlxEngineGroupSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 6) {
                Image(systemName: "waveform.badge.magnifyingglass")
                    .foregroundStyle(self.theme.palette.accent)
                Text("MLX Engine".loc)
                    .font(.headline)
                    .foregroundStyle(.primary)
                Text("(内置独立引擎 · 完全离线)".loc)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                Spacer()
                if self.activeModel.isMlxEngineModel,
                    let card = self.selectedMlxCard
                {
                    self.miniBadge(text: card.title, color: Color.fluidGreen)
                }
            }

            VStack(alignment: .leading, spacing: 12) {
                // Context / hotwords (official Qwen3-ASR "Vocabulary: ..." system prompt).
                // Qwen3-ASR 模型卡专属；其余模型卡（Parakeet/Nemotron/GLM/Fun/Whisper）不使用。
                VStack(alignment: .leading, spacing: 3) {
                    Text("词汇 / 热词 (Vocabulary · 仅 Qwen3-ASR 卡生效)".loc)
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(.secondary)
                    TextField(
                        "e.g. qwen3, 千问, Qwen3-ASR",
                        text: Binding(
                            get: { self.settings.qwen3ContextWords },
                            set: { self.settings.qwen3ContextWords = $0 }
                        )
                    )
                    .textFieldStyle(.roundedBorder)
                    .font(.system(size: 11))
                    .frame(maxWidth: 380)
                    Text("以官方 Vocabulary 提示词注入，显著提升专有名词/品牌词识别（如 qwen3 → 千万三）。".loc)
                        .font(.system(size: 10))
                        .foregroundStyle(.secondary)
                }

                Divider().opacity(0.3)

                Text("模型卡：每张卡独立下载 / 卸载 · 点击卡片即切换".loc)
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(.secondary)

                VStack(spacing: 6) {
                    ForEach(MlxSttCatalog.cardsByRecommendation) { card in
                        self.mlxCardRow(card)
                    }
                }
            }
            .padding(12)
            .background(
                RoundedRectangle(cornerRadius: 12)
                    .fill(Color(nsColor: .controlBackgroundColor).opacity(0.6))
                    .overlay(
                        RoundedRectangle(cornerRadius: 12)
                            .stroke(Color.secondary.opacity(0.14), lineWidth: 1)
                    )
            )
        }
    }

    /// One MLX model card row: select / download / uninstall per card.
    private func mlxCardRow(_ card: MlxSttCard) -> some View {
        let isSelected = self.settings.selectedMlxSttCardID == card.pathID
        let isInstalled: Bool
        switch card.family {
        case .parakeetTdt:
            isInstalled = ParakeetMlxEngine.modelsExist(modelID: card.family.rawValue, variantID: card.id)
        case .nemotronAsr:
            isInstalled = NemotronMlxEngine.modelsExist(modelID: card.family.rawValue, variantID: card.id)
        case .glmAsr:
            isInstalled = GlmMlxEngine.modelsExist(modelID: card.family.rawValue, variantID: card.id)
        case .funAsr:
            isInstalled = FunMlxEngine.modelsExist(modelID: card.family.rawValue, variantID: card.id)
        case .whisperMlx:
            isInstalled = WhisperMlxEngine.modelsExist(modelID: card.family.rawValue, variantID: card.id)
        case .qwen3Asr:
            isInstalled = Qwen3MlxEngine.modelsExist(modelID: card.family.rawValue, variantID: card.id)
        }
        let isDownloading = self.viewModel.downloadingModel == .qwen3Asr
            && self.viewModel.isMlxCardDownloading(card)
        // Exclusive: an MLX card is only "selected" while the app is actually
        // running on the MLX engine (qwen3-asr / whisper). When the user picks
        // Apple or a cloud model, the MLX card list must NOT show any card as
        // active/selected — the single selectedSpeechModel drives the routing.
        let isMlxEngineActive = self.activeModel.isMlxEngineModel
        let isActive = isMlxEngineActive && isSelected
        let isCardHighlighted = isMlxEngineActive && isSelected

        return HStack(spacing: 10) {
            Image(systemName: "waveform.badge.mic")
                .font(.system(size: 13))
                .foregroundStyle(isCardHighlighted ? self.theme.palette.accent : .secondary)
                .frame(width: 20)

            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 6) {
                    Text(card.title)
                        .font(.system(size: 12, weight: isCardHighlighted ? .semibold : .regular))
                    let rec = MlxSttCatalog.recommendation(for: card)
                    self.miniBadge(
                        text: rec.label,
                        color: rec == .notRecommended ? .orange : (rec == .mustHave ? Color.fluidGreen : rec == .recommended ? .blue : .secondary)
                    )
                    self.miniBadge(text: card.family.displayName, color: .blue)
                    if let quant = card.quantLabel {
                        self.miniBadge(text: quant, color: .orange)
                    }
                    self.miniBadge(text: card.sizeDescription, color: .secondary)
                    if isActive {
                        self.miniBadge(text: "使用中".loc, color: Color.fluidGreen)
                    }
                }
                Text(card.footnote)
                    .font(.system(size: 10))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }

            Spacer()

            if isDownloading {
                VStack(alignment: .trailing, spacing: 3) {
                    ProgressView(value: self.viewModel.downloadProgress)
                        .frame(width: 70)
                    Text("\(Int(self.viewModel.downloadProgress * 100))%")
                        .font(.system(size: 9))
                        .monospacedDigit()
                        .foregroundStyle(.secondary)
                }
            } else if !isInstalled {
                Button {
                    self.viewModel.downloadMlxCard(card)
                } label: {
                    Image(systemName: "arrow.down.circle")
                        .font(.system(size: 13))
                        .foregroundStyle(self.theme.palette.accent)
                }
                .buttonStyle(.plain)
                .help("下载此量化模型卡".loc)
            } else {
                HStack(spacing: 10) {
                    Button {
                        self.viewModel.uninstallMlxCard(card)
                    } label: {
                        Image(systemName: "trash")
                            .font(.system(size: 12))
                            .foregroundStyle(.secondary.opacity(0.7))
                    }
                    .buttonStyle(.plain)
                    .help("卸载此模型卡以释放磁盘空间".loc)

                    Button {
                        self.viewModel.selectMlxCard(card)
                    } label: {
                        if isActive {
                            Image(systemName: "checkmark.circle.fill")
                                .foregroundStyle(self.theme.palette.accent)
                                .font(.system(size: 16))
                        } else {
                            Circle()
                                .strokeBorder(Color.secondary.opacity(0.35), lineWidth: 1.5)
                                .frame(width: 16, height: 16)
                        }
                    }
                    .buttonStyle(.plain)
                    .help("使用此模型卡".loc)
                }
            }
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 6)
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(isActive ? self.theme.palette.accent.opacity(0.08) : Color.secondary.opacity(0.05))
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(isActive ? self.theme.palette.accent.opacity(0.4) : Color.secondary.opacity(0.12), lineWidth: 1)
                )
        )
        .contentShape(Rectangle())
        .onTapGesture {
            // 整行可点击：点击卡片即切换为该模型卡（引擎为 MLX，无需嵌套选择）。
            self.viewModel.selectMlxCard(card)
        }
    }

    // MARK: - Model Card Component

    private func modelCard(for model: SpeechModel) -> some View {
        let isSelected = self.activeModel == model
        let isInstalled = model.isInstalled
        let isDownloading = self.viewModel.downloadingModel == model

        return VStack(spacing: 0) {
            // Main Row
            HStack(spacing: 12) {
                Image(systemName: self.iconForModel(model))
                    .font(.system(size: 18))
                    .foregroundStyle(isSelected ? self.theme.palette.accent : .secondary)
                    .frame(width: 28, height: 28)

                VStack(alignment: .leading, spacing: 3) {
                    HStack(spacing: 6) {
                        Text(self.displayNameForModel(model))
                            .font(.system(size: 13, weight: isSelected ? .semibold : .regular))
                            .foregroundStyle(.primary)

                        // Badges
                        if model == .appleSpeech || model == .appleSpeechAnalyzer {
                            self.miniBadge(text: "Built-in".loc, color: .blue)
                        } else if model.isCloudModel {
                            self.miniBadge(text: "Cloud / LAN".loc, color: .purple)
                        } else {
                            self.miniBadge(text: model.downloadSize, color: .secondary)
                        }

                        if model == .appleSpeechAnalyzer {
                            self.miniBadge(text: "Recommended".loc, color: Color.fluidGreen)
                        }
                    }

                    Text(self.descriptionForModel(model))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                }

                Spacer()

                // Actions: Select / Download / Progress
                self.modelActionView(for: model, isSelected: isSelected, isInstalled: isInstalled, isDownloading: isDownloading)
            }
            .padding(12)

            // Inline Configuration for Selected Cloud Engine
            if isSelected && model.isCloudModel {
                Divider().opacity(0.3).padding(.horizontal, 10)
                self.inlineCloudConfigView(for: model)
                    .padding(12)
                    .background(Color.secondary.opacity(0.04))
            }
        }
        .background(
            RoundedRectangle(cornerRadius: 10)
                .fill(isSelected ? self.theme.palette.accent.opacity(0.06) : Color(nsColor: .controlBackgroundColor).opacity(0.6))
                .overlay(
                    RoundedRectangle(cornerRadius: 10)
                        .stroke(isSelected ? self.theme.palette.accent.opacity(0.45) : Color.secondary.opacity(0.14), lineWidth: 1)
                )
        )
        .contentShape(Rectangle())
        .onTapGesture {
            if isInstalled && !isSelected {
                withAnimation(.easeInOut(duration: 0.15)) {
                    self.viewModel.activateSpeechModel(model)
                }
            }
        }
    }

    @ViewBuilder
    private func modelActionView(
        for model: SpeechModel, isSelected: Bool, isInstalled: Bool, isDownloading: Bool
    ) -> some View {
        if isDownloading {
            VStack(alignment: .trailing, spacing: 4) {
                HStack(spacing: 6) {
                    ProgressView(value: self.viewModel.downloadProgress)
                        .frame(width: 80)
                    Text("\(Int(self.viewModel.downloadProgress * 100))%")
                        .font(.caption2)
                        .monospacedDigit()
                }
                Button("Cancel".loc) {
                    self.viewModel.cancelSpeechModelDownload()
                }
                .buttonStyle(.plain)
                .font(.caption2)
                .foregroundStyle(.red)
            }
        } else if !isInstalled {
            Button {
                self.viewModel.downloadSpeechModel(model)
            } label: {
                HStack(spacing: 4) {
                    Image(systemName: "arrow.down.circle")
                    Text("Download".loc)
                }
                .font(.system(size: 12, weight: .medium))
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
        } else {
            HStack(spacing: 12) {
                // Delete / Uninstall button for downloadable offline models
                if !model.isCloudModel && model != .appleSpeech && model != .appleSpeechAnalyzer && model != .qwen3Asr {
                    Button {
                        Task {
                            await self.viewModel.deleteModel(model)
                        }
                    } label: {
                        Image(systemName: "trash")
                            .font(.system(size: 13))
                            .foregroundStyle(.secondary.opacity(0.7))
                    }
                    .buttonStyle(.plain)
                    .help("卸载此模型以释放磁盘空间".loc)
                }

                Button {
                    withAnimation(.easeInOut(duration: 0.15)) {
                        self.viewModel.activateSpeechModel(model)
                    }
                } label: {
                    if isSelected {
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundStyle(self.theme.palette.accent)
                            .font(.system(size: 18))
                    } else {
                        Circle()
                            .strokeBorder(Color.secondary.opacity(0.35), lineWidth: 1.5)
                            .frame(width: 18, height: 18)
                    }
                }
                .buttonStyle(.plain)
            }
        }
    }

    private func miniBadge(text: String, color: Color) -> some View {
        Text(text)
            .font(.system(size: 9, weight: .bold))
            .padding(.horizontal, 5)
            .padding(.vertical, 1.5)
            .background(Capsule().fill(color.opacity(0.15)))
            .foregroundStyle(color)
    }

    // MARK: - Inline Configuration View for Selected Cloud Engine
    @ViewBuilder
    private func inlineCloudConfigView(for model: SpeechModel) -> some View {
        VStack(spacing: 10) {
            if model == .cloudOllama {
                HStack(alignment: .center) {
                    Text("Ollama Base URL".loc)
                        .font(.subheadline)
                        .frame(width: 140, alignment: .leading)
                    TextField("http://192.168.7.136:11434", text: self.$settings.cloudSTTOllamaBaseURL)
                        .textFieldStyle(.roundedBorder)
                }
                HStack(alignment: .center) {
                    Text("Model Name".loc)
                        .font(.subheadline)
                        .frame(width: 140, alignment: .leading)
                    TextField("e.g. qwen3-asr", text: self.$settings.cloudSTTOllamaModel)
                        .textFieldStyle(.roundedBorder)
                }

                // Ollama model list (scan the server's /api/tags)
                HStack(alignment: .center) {
                    Text("Available Models".loc)
                        .font(.subheadline)
                        .frame(width: 140, alignment: .leading)

                    if self.isFetchingOllamaModels {
                        ProgressView()
                            .controlSize(.small)
                            .padding(.leading, 4)
                    } else if self.fetchedOllamaModels.isEmpty, self.ollamaFetchErrorMessage == nil {
                        Text("Click Scan Models to load the list from your server".loc)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    } else {
                        Picker("", selection: self.$settings.cloudSTTOllamaModel) {
                            ForEach(self.fetchedOllamaModels, id: \.self) { name in
                                Text(name).tag(name)
                            }
                        }
                        .labelsHidden()
                        .pickerStyle(.menu)
                        .frame(maxWidth: .infinity, alignment: .leading)
                    }

                    Spacer()

                    Button {
                        self.fetchOllamaModels()
                    } label: {
                        HStack(spacing: 4) {
                            if self.isFetchingOllamaModels {
                                ProgressView().controlSize(.small)
                            } else {
                                Image(systemName: "arrow.clockwise")
                            }
                            Text("Scan Models".loc)
                        }
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    .disabled(self.isFetchingOllamaModels)
                }

                if let message = self.ollamaFetchErrorMessage {
                    HStack {
                        Text(message)
                            .font(.caption)
                            .foregroundStyle(.red.opacity(0.85))
                        Spacer()
                    }
                }
            } else if model == .cloudOpenAI {
                HStack(alignment: .center) {
                    Text("OpenAI API Key".loc)
                        .font(.subheadline)
                        .frame(width: 140, alignment: .leading)
                    SecureField("sk-...", text: self.$settings.cloudSTTOpenAIAPIKey)
                        .textFieldStyle(.roundedBorder)
                }
                HStack(alignment: .center) {
                    Text("Model".loc)
                        .font(.subheadline)
                        .frame(width: 140, alignment: .leading)
                    TextField("whisper-1", text: self.$settings.cloudSTTOpenAIModel)
                        .textFieldStyle(.roundedBorder)
                }
            } else if model == .cloudGroq {
                HStack(alignment: .center) {
                    Text("Groq API Key".loc)
                        .font(.subheadline)
                        .frame(width: 140, alignment: .leading)
                    SecureField("gsk_...", text: self.$settings.cloudSTTGroqAPIKey)
                        .textFieldStyle(.roundedBorder)
                }
                HStack(alignment: .center) {
                    Text("Model".loc)
                        .font(.subheadline)
                        .frame(width: 140, alignment: .leading)
                    TextField("whisper-large-v3-turbo", text: self.$settings.cloudSTTGroqModel)
                        .textFieldStyle(.roundedBorder)
                }
            } else if model == .cloudOpenRouter {
                HStack(alignment: .center) {
                    Text("OpenRouter API Key".loc)
                        .font(.subheadline)
                        .frame(width: 140, alignment: .leading)
                    SecureField("sk-or-...", text: self.$settings.cloudSTTOpenRouterAPIKey)
                        .textFieldStyle(.roundedBorder)
                }
                HStack(alignment: .center) {
                    Text("Model".loc)
                        .font(.subheadline)
                        .frame(width: 140, alignment: .leading)
                    TextField("e.g. openai/whisper-large-v3-turbo", text: self.$settings.cloudSTTOpenRouterModel)
                        .textFieldStyle(.roundedBorder)
                }

                // OpenRouter model list (GET /api/v1/models, OpenAI format)
                HStack(alignment: .center) {
                    Text("Available Models".loc)
                        .font(.subheadline)
                        .frame(width: 140, alignment: .leading)

                    if self.isFetchingOpenRouterModels {
                        ProgressView()
                            .controlSize(.small)
                            .padding(.leading, 4)
                    } else if self.fetchedOpenRouterModels.isEmpty, self.openRouterFetchErrorMessage == nil {
                        Text("Click Scan Models to load the list from OpenRouter".loc)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    } else {
                        Picker("", selection: self.$settings.cloudSTTOpenRouterModel) {
                            ForEach(self.fetchedOpenRouterModels, id: \.self) { name in
                                Text(name).tag(name)
                            }
                        }
                        .labelsHidden()
                        .pickerStyle(.menu)
                        .frame(maxWidth: .infinity, alignment: .leading)
                    }

                    Spacer()

                    Button {
                        self.fetchOpenRouterModels()
                    } label: {
                        HStack(spacing: 4) {
                            if self.isFetchingOpenRouterModels {
                                ProgressView().controlSize(.small)
                            } else {
                                Image(systemName: "arrow.clockwise")
                            }
                            Text("Scan Models".loc)
                        }
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    .disabled(self.isFetchingOpenRouterModels)
                }

                if let message = self.openRouterFetchErrorMessage {
                    HStack {
                        Text(message)
                            .font(.caption)
                            .foregroundStyle(.red.opacity(0.85))
                        Spacer()
                    }
                }
            } else if model == .cloudCustom {
                HStack(alignment: .center) {
                    Text("Endpoint URL".loc)
                        .font(.subheadline)
                        .frame(width: 140, alignment: .leading)
                    TextField("https://api.example.com/v1/audio/transcriptions", text: self.$settings.cloudSTTCustomBaseURL)
                        .textFieldStyle(.roundedBorder)
                }
                HStack(alignment: .center) {
                    Text("API Key".loc)
                        .font(.subheadline)
                        .frame(width: 140, alignment: .leading)
                    SecureField("Optional API Key...", text: self.$settings.cloudSTTCustomAPIKey)
                        .textFieldStyle(.roundedBorder)
                }
            }

            // Language & Test Button Row
            HStack(alignment: .center) {
                Text("Target Language".loc)
                    .font(.subheadline)
                    .frame(width: 140, alignment: .leading)
                Picker("", selection: self.$settings.cloudSTTLanguage) {
                    Text("Auto Detect".loc).tag("auto")
                    Text("Chinese (Simplified)".loc).tag("zh")
                    Text("English".loc).tag("en")
                    Text("Japanese".loc).tag("ja")
                    Text("Korean".loc).tag("ko")
                }
                .labelsHidden()
                .frame(width: 150)

                Spacer()

                Button {
                    self.testConnection()
                } label: {
                    HStack(spacing: 4) {
                        if self.isTestingConnection {
                            ProgressView().controlSize(.small)
                        } else {
                            Image(systemName: "network")
                        }
                        Text(self.isTestingConnection ? "Testing...".loc : "Test Connection".loc)
                    }
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .disabled(self.isTestingConnection)
            }

            if let message = self.connectionTestMessage {
                HStack {
                    Spacer()
                    Text(message)
                        .font(.caption)
                        .foregroundStyle(self.connectionTestSuccess == true ? Color.fluidGreen : .red)
                }
            }

            // ASR Prompt Field
            VStack(alignment: .leading, spacing: 3) {
                Text("ASR Prompt (Optional / Vocabulary Biasing)".loc)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                TextField("e.g. Specialized terminology, proper nouns, or spelling cues".loc, text: self.$settings.cloudSTTPrompt)
                    .textFieldStyle(.roundedBorder)
                    .font(.subheadline)
            }
        }
    }

    // MARK: - 5. Smart Fallback Section
    private var smartFallbackSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Reliability & Fallback".loc)
                .font(.headline)
                .foregroundStyle(.primary)

            VStack(spacing: 12) {
                HStack(alignment: .top) {
                    VStack(alignment: .leading, spacing: 3) {
                        Text("macOS 本地语音智能兜底 (Apple Speech)".loc)
                            .font(.subheadline)
                            .fontWeight(.medium)
                        Text("当云端或 Ollama 响应超时或发生异常时，自动无缝切换到系统内置本地引擎转录，保障录音 100% 不丢失。".loc)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                    Toggle("", isOn: self.$settings.cloudSTTAutoFallback)
                        .labelsHidden()
                        .toggleStyle(.switch)
                }

                Divider().opacity(0.3)

                HStack {
                    Text("响应超时时间".loc)
                        .font(.subheadline)
                    Spacer()
                    Picker("", selection: self.$settings.cloudSTTTimeoutSeconds) {
                        Text("4 秒 (极速判定)".loc).tag(4.0)
                        Text("6 秒 (推荐)".loc).tag(6.0)
                        Text("8 秒 (标准)".loc).tag(8.0)
                        Text("12 秒 (宽松)".loc).tag(12.0)
                    }
                    .labelsHidden()
                    .frame(width: 140)
                }
            }
            .padding(14)
            .background(
                RoundedRectangle(cornerRadius: 12)
                    .fill(Color(nsColor: .controlBackgroundColor))
                    .overlay(
                        RoundedRectangle(cornerRadius: 12)
                            .stroke(Color.secondary.opacity(0.18), lineWidth: 1)
                    )
            )
        }
    }

    // MARK: - 6. Tip Section
    private var tipSection: some View {
        HStack {
            Image(systemName: "character.book.closed.fill")
                .foregroundStyle(.secondary)
            Text("Tip: You can also configure word replacements and formatting rules in Custom Dictionary.".loc)
                .font(.caption)
                .foregroundStyle(.secondary)
            Spacer()
        }
        .padding(.vertical, 4)
    }

    // MARK: - Helpers & Display Names
    private func iconForModel(_ model: SpeechModel) -> String {
        switch model {
        case .appleSpeech, .appleSpeechAnalyzer: return "apple.logo"
        case .cloudOllama: return "network"
        case .cloudOpenAI, .cloudGroq, .cloudOpenRouter, .cloudCustom: return "cloud.fill"
        default: return "cpu.fill"
        }
    }

    private func displayNameForModel(_ model: SpeechModel) -> String {
        switch model {
        case .appleSpeech: return "Apple Speech (经典内置引擎)".loc
        case .appleSpeechAnalyzer: return "Apple Speech Analyzer (macOS 26+ 现代流式)".loc
        case .cloudOllama: return "Ollama ASR (局域网 / 本地服务器)".loc
        case .cloudOpenAI: return "OpenAI Cloud STT (Whisper API)".loc
        case .cloudGroq: return "Groq Cloud STT (极速 LPUs)".loc
        case .cloudOpenRouter: return "OpenRouter Cloud STT".loc
        case .cloudCustom: return "Custom Cloud STT (自定义接口)".loc
        case .qwen3Asr: return "MLX 语音引擎 (本地离线)".loc
        case .whisperTiny: return "Whisper Tiny (超轻量)".loc
        case .whisperBase: return "Whisper Base (基础模型)".loc
        case .whisperSmall: return "Whisper Small (平衡推荐)".loc
        case .whisperMedium: return "Whisper Medium (高精度)".loc
        case .whisperLargeTurbo: return "Whisper Large Turbo (推荐离线)".loc
        case .whisperLarge: return "Whisper Large V3 (终极精度)".loc
        }
    }

    private func descriptionForModel(_ model: SpeechModel) -> String {
        switch model {
        case .appleSpeech:
            return "macOS 系统经典内置听写，零内存占用，完全离线，开箱即用。".loc
        case .appleSpeechAnalyzer:
            return "macOS 26+ 采用的新一代神经流式识别引擎，延迟更低、分词更精准。".loc
        case .cloudOllama:
            return "连接到本地或局域网 Ollama 服务（如 Qwen3-ASR、SenseVoice），免除本机显存负担。".loc
        case .cloudOpenAI:
            return "官方 OpenAI Whisper 云端 API，支持 99+ 种语言极高精度转录。".loc
        case .cloudGroq:
            return "由 Groq LPU 硬件加速驱动，几乎零延迟的超高速云端 Whisper 识别。".loc
        case .cloudOpenRouter:
            return "通过 OpenRouter 聚合网关调用云端语音大模型。".loc
        case .cloudCustom:
            return "自定义兼容 OpenAI 标准格式的语音识别服务地址。".loc
        case .qwen3Asr:
            return "内置 MLX 语音引擎：Qwen3-ASR / Parakeet / Nemotron / GLM-ASR / Fun-ASR 多模型卡，纯本地完全离线。".loc
        case .whisperTiny:
            return "体积仅 ~44MB，极低资源占用，适合省电或老款机型。".loc
        case .whisperBase:
            return "体积 ~81MB，速度与精度的良好平衡。".loc
        case .whisperSmall:
            return "体积 ~257MB，日常中文与英文听写的高性价比选择。".loc
        case .whisperMedium:
            return "体积 ~793MB，高精度语音转录，适合复杂词汇与专业术语。".loc
        case .whisperLargeTurbo:
            return "体积 ~845MB，Whisper 官方优化版大模型，兼具顶尖精度与极快速度。".loc
        case .whisperLarge:
            return "体积 ~1.55GB，最高精度的全尺寸 Whisper 离线模型。".loc
        }
    }

    private func subtitleForModel(_ model: SpeechModel) -> String {
        switch model {
        case .appleSpeech: return "macOS 经典内置听写引擎 (SFSpeechRecognizer)".loc
        case .appleSpeechAnalyzer: return "macOS 26+ 现代神经流式引擎 (SpeechAnalyzer)".loc
        case .cloudOllama: return "\(self.settings.cloudSTTOllamaBaseURL) (\(self.settings.cloudSTTOllamaModel))"
        case .cloudOpenAI: return "OpenAI Cloud STT (\(self.settings.cloudSTTOpenAIModel))"
        case .cloudGroq: return "Groq Cloud STT (\(self.settings.cloudSTTGroqModel))"
        case .qwen3Asr:
            // The hero card reflects the currently selected MLX model card.
            if let card = MlxSttCatalog.card(pathID: self.settings.selectedMlxSttCardID) {
                return "\(card.title) · \(card.sizeDescription)"
            }
            return "本地离线神经网络引擎 (\(model.downloadSize))".loc
        default: return "本地离线神经网络引擎 (\(model.downloadSize))".loc
        }
    }

    /// Fetch the model list from an Ollama server (`/api/tags`) with a short
    /// timeout and visible error reporting.
    private func fetchOllamaModels() {
        let rawBase = self.settings.cloudSTTOllamaBaseURL
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        let base = rawBase.isEmpty ? "http://localhost:11434" : rawBase

        self.isFetchingOllamaModels = true
        self.ollamaFetchErrorMessage = nil
        Task {
            defer {
                Task { @MainActor in
                    self.isFetchingOllamaModels = false
                }
            }
            do {
                guard let url = URL(string: "\(base)/api/tags") else {
                    throw NSError(
                        domain: "NativeVoiceEngineSettingsView",
                        code: -1,
                        userInfo: [NSLocalizedDescriptionKey: "Invalid Ollama base URL: \(base)"]
                    )
                }
                var request = URLRequest(url: url)
                request.timeoutInterval = 8
                let (data, response) = try await URLSession.shared.data(for: request)
                if let http = response as? HTTPURLResponse, http.statusCode >= 400 {
                    throw NSError(
                        domain: "NativeVoiceEngineSettingsView",
                        code: http.statusCode,
                        userInfo: [NSLocalizedDescriptionKey: "HTTP \(http.statusCode)"]
                    )
                }
                if let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                   let modelsArray = json["models"] as? [[String: Any]]
                {
                    let names = modelsArray.compactMap { $0["name"] as? String }
                    await MainActor.run {
                        self.fetchedOllamaModels = names
                        if names.isEmpty {
                            self.ollamaFetchErrorMessage =
                                "No models found. Make sure the Ollama server is running and has models pulled.".loc
                        }
                    }
                } else {
                    await MainActor.run {
                        self.fetchedOllamaModels = []
                        self.ollamaFetchErrorMessage =
                            "Unexpected response from %@. Is this really an Ollama server?".locFormat(url.host ?? base)
                    }
                }
            } catch {
                DebugLogger.shared.error("Failed to fetch Ollama models: \(error)", source: "NativeVoiceEngineSettingsView")
                await MainActor.run {
                    self.fetchedOllamaModels = []
                    self.ollamaFetchErrorMessage =
                        "Could not reach Ollama at %@. Please check the address and that the server is running.".locFormat(base)
                        + " (\(error.localizedDescription))"
                }
            }
        }
    }

    /// Fetch the model list from OpenRouter (`/api/v1/models`, OpenAI format)
    /// with a short timeout and visible error reporting. Typical response:
    /// { "data": [{ "id": "openai/whisper-large-v3-turbo", ... }, ...] }
    private func fetchOpenRouterModels() {
        let rawBase = self.settings.cloudSTTOpenRouterBaseURL
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        let base = rawBase.isEmpty ? "https://openrouter.ai/api/v1" : rawBase

        self.isFetchingOpenRouterModels = true
        self.openRouterFetchErrorMessage = nil
        Task {
            defer {
                Task { @MainActor in
                    self.isFetchingOpenRouterModels = false
                }
            }
            do {
                guard let url = URL(string: "\(base)/models?output_modalities=transcription") else {
                    throw NSError(
                        domain: "NativeVoiceEngineSettingsView",
                        code: -1,
                        userInfo: [NSLocalizedDescriptionKey: "Invalid OpenRouter base URL: \(base)"]
                    )
                }
                var request = URLRequest(url: url)
                request.timeoutInterval = 10
                let apiKey = self.settings.cloudSTTOpenRouterAPIKey
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                if !apiKey.isEmpty {
                    request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
                }
                let (data, response) = try await URLSession.shared.data(for: request)
                if let http = response as? HTTPURLResponse, http.statusCode >= 400 {
                    throw NSError(
                        domain: "NativeVoiceEngineSettingsView",
                        code: http.statusCode,
                        userInfo: [NSLocalizedDescriptionKey: "HTTP \(http.statusCode)"]
                    )
                }
                if let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                   let modelsArray = json["data"] as? [[String: Any]]
                {
                    // The endpoint already filters to transcription-capable
                    // models (output_modalities=transcription) — same source as
                    // the site's STT page. Sort with the curated/recommended
                    // ones first, then the rest alphabetically.
                    let names = modelsArray.compactMap { $0["id"] as? String }
                    let unique = Array(Set(names)).sorted()
                    let curatedFirst = Self.curatedOpenRouterSTTModels.filter { unique.contains($0) }
                        + unique.filter { !Self.curatedOpenRouterSTTModels.contains($0) }
                    await MainActor.run {
                        self.fetchedOpenRouterModels = curatedFirst
                        if unique.isEmpty {
                            // Fall back to the curated list so the user always
                            // has a working STT model to pick.
                            self.fetchedOpenRouterModels = Self.curatedOpenRouterSTTModels
                            self.openRouterFetchErrorMessage =
                                "No ASR models detected; showing the recommended list.".loc
                        }
                    }
                } else {
                    await MainActor.run {
                        self.fetchedOpenRouterModels = []
                        self.openRouterFetchErrorMessage =
                            "Unexpected response from OpenRouter. Expected a JSON `data` array.".loc
                    }
                }
            } catch {
                DebugLogger.shared.error("Failed to fetch OpenRouter models: \(error)", source: "NativeVoiceEngineSettingsView")
                await MainActor.run {
                    self.fetchedOpenRouterModels = []
                    self.openRouterFetchErrorMessage =
                        "Could not reach OpenRouter at %@. Please check the address and that the server is running.".locFormat(base)
                        + " (\(error.localizedDescription))"
                }
            }
        }
    }

    private func testConnection() {        self.isTestingConnection = true
        self.connectionTestMessage = nil
        self.connectionTestSuccess = nil

        let type: CloudSTTType
        switch self.activeModel {
        case .cloudOllama: type = .ollama
        case .cloudOpenAI: type = .openAI
        case .cloudGroq: type = .groq
        case .cloudOpenRouter: type = .openRouter
        default: type = .custom
        }

        let provider = CloudTranscriptionProvider(type: type, settings: self.settings)

        Task {
            let samples = Array(repeating: Float(0.0), count: 16000)
            do {
                _ = try await provider.transcribe(samples)
                await MainActor.run {
                    self.isTestingConnection = false
                    self.connectionTestSuccess = true
                    self.connectionTestMessage = "Connection verified".loc
                }
            } catch {
                await MainActor.run {
                    self.isTestingConnection = false
                    self.connectionTestSuccess = false
                    self.connectionTestMessage = "\("Connection failed".loc): \(error.localizedDescription)"
                }
            }
        }
    }

    // MARK: - Download Acceleration Section
    private var downloadAccelerationSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                Image(systemName: "bolt.horizontal.circle.fill")
                    .foregroundStyle(.yellow)
                    .font(.system(size: 15))
                Text("模型下载加速源".loc)
                    .font(.subheadline)
                    .fontWeight(.medium)

                Spacer()

                Picker("", selection: self.$settings.huggingFaceMirror) {
                    ForEach(SettingsStore.HuggingFaceMirror.allCases) { mirror in
                        Text(mirror.displayName.loc).tag(mirror)
                    }
                }
                .pickerStyle(.menu)
                .controlSize(.small)
                .frame(maxWidth: 270)
            }

            if self.settings.huggingFaceMirror == .custom {
                HStack(spacing: 8) {
                    Text("自定义端点:".loc)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    TextField("https://your-mirror-host.com", text: self.$settings.customHuggingFaceMirrorURL)
                        .textFieldStyle(.roundedBorder)
                        .controlSize(.small)
                }
                .padding(.top, 2)
            }

            Text("默认使用 Hugging Face 官方源（huggingface.co）。若所在网络访问受限，可切换到国内镜像（hf-mirror.com）或其他自定义镜像源——镜像站部分大模型无缓存，可能回源到官网。".loc)
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        .padding(12)
        .background(
            RoundedRectangle(cornerRadius: 10)
                .fill(Color(nsColor: .controlBackgroundColor))
                .overlay(
                    RoundedRectangle(cornerRadius: 10)
                        .stroke(Color.secondary.opacity(0.15), lineWidth: 1)
                )
        )
    }

    // MARK: - Context & Memory Estimator
    private var contextMemoryEstimatorSection: some View {
        let estimate = self.settings.estimateMemoryBreakdown(model: self.activeModel)
        return VStack(alignment: .leading, spacing: 10) {
            // Header Row
            HStack {
                HStack(spacing: 6) {
                    Image(systemName: "memorychip")
                        .foregroundStyle(self.theme.palette.accent)
                        .font(.system(size: 15))
                    Text("本地模型上下文与内存预估 (Context & RAM)".loc)
                        .font(.subheadline)
                        .fontWeight(.medium)
                }

                Spacer()

                // Pressure Badge
                HStack(spacing: 4) {
                    Circle()
                        .fill(estimate.pressure == .safe ? Color.fluidGreen : (estimate.pressure == .moderate ? Color.orange : Color.red))
                        .frame(width: 7, height: 7)
                    Text("\("负载:".loc) \(estimate.pressure.rawValue)")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(estimate.pressure == .safe ? Color.fluidGreen : (estimate.pressure == .moderate ? Color.orange : Color.red))
                }
                .padding(.horizontal, 8)
                .padding(.vertical, 3)
                .background(
                    Capsule().fill((estimate.pressure == .safe ? Color.fluidGreen : (estimate.pressure == .moderate ? Color.orange : Color.red)).opacity(0.12))
                )
            }

            // Context Length Segmented Picker
            VStack(alignment: .leading, spacing: 5) {
                HStack {
                    Text("单次识别最长时长:".loc)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer()
                    if let selectedDuration = SettingsStore.ASRContextDuration(rawValue: self.settings.asrContextTokenLimit) {
                        Text(selectedDuration.subtitle)
                            .font(.caption2)
                            .foregroundStyle(self.theme.palette.accent)
                    }
                }

                Picker("", selection: self.$settings.asrContextTokenLimit) {
                    ForEach(SettingsStore.ASRContextDuration.allCases) { duration in
                        Text(duration.displayName).tag(duration.rawValue)
                    }
                }
                .pickerStyle(.segmented)
                .controlSize(.small)
            }

            // Detailed Memory Bar Breakdown
            VStack(spacing: 6) {
                HStack {
                    HStack(spacing: 8) {
                        HStack(spacing: 3) {
                            Circle().fill(Color.blue).frame(width: 5, height: 5)
                            Text("\("权重:".loc) \(String(format: "%.1f", estimate.modelWeightGB))G")
                                .font(.system(size: 11))
                                .foregroundStyle(.secondary)
                        }
                        HStack(spacing: 3) {
                            Circle().fill(Color.purple).frame(width: 5, height: 5)
                            Text("\("KV缓存:".loc) \(String(format: "%.2f", estimate.contextKVCacheGB))G")
                                .font(.system(size: 11))
                                .foregroundStyle(.secondary)
                        }
                        HStack(spacing: 3) {
                            Circle().fill(Color.gray.opacity(0.7)).frame(width: 5, height: 5)
                            Text("\("开销:".loc) \(String(format: "%.1f", estimate.runtimeOverheadGB))G")
                                .font(.system(size: 11))
                                .foregroundStyle(.secondary)
                        }
                    }

                    Spacer()

                    Text("\("预计:".loc) \(String(format: "%.2f", estimate.totalGB))G / \(String(format: "%.0f", estimate.totalDeviceRAMGB))G RAM · 可用 \(String(format: "%.1f", estimate.availableRAMGB))G")
                        .font(.system(size: 11, weight: .bold))
                        .foregroundStyle(estimate.pressure == .safe ? Color.fluidGreen : Color.primary)
                }

                // Visual Memory Usage Bar
                GeometryReader { geo in
                    let totalWidth = geo.size.width
                    let totalDeviceRAM = max(estimate.totalDeviceRAMGB, 8.0)
                    let weightRatio = CGFloat(estimate.modelWeightGB / totalDeviceRAM)
                    let kvRatio = CGFloat(estimate.contextKVCacheGB / totalDeviceRAM)
                    let overheadRatio = CGFloat(estimate.runtimeOverheadGB / totalDeviceRAM)

                    ZStack(alignment: .leading) {
                        RoundedRectangle(cornerRadius: 3)
                            .fill(Color.secondary.opacity(0.12))
                            .frame(height: 6)

                        HStack(spacing: 1) {
                            RoundedRectangle(cornerRadius: 2)
                                .fill(Color.blue)
                                .frame(width: max(2, totalWidth * weightRatio), height: 6)
                            RoundedRectangle(cornerRadius: 2)
                                .fill(Color.purple)
                                .frame(width: max(2, totalWidth * kvRatio), height: 6)
                            RoundedRectangle(cornerRadius: 2)
                                .fill(Color.gray.opacity(0.7))
                                .frame(width: max(2, totalWidth * overheadRatio), height: 6)
                        }
                    }
                }
                .frame(height: 6)
            }
            .padding(8)
            .background(
                RoundedRectangle(cornerRadius: 6, style: .continuous)
                    .fill(Color.secondary.opacity(0.06))
            )
        }
        .padding(12)
        .background(
            RoundedRectangle(cornerRadius: 10)
                .fill(Color(nsColor: .controlBackgroundColor))
                .overlay(
                    RoundedRectangle(cornerRadius: 10)
                        .stroke(Color.secondary.opacity(0.15), lineWidth: 1)
                )
        )
    }
}
