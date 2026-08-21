//
//  NativeVoiceEngineSettingsView.swift
//  FluidVoice
//
//  Modern, comprehensive native SwiftUI Voice Engine Settings View adhering to Apple HIG.
//

import SwiftUI

typealias SpeechModel = SettingsStore.SpeechModel

enum ModelCategoryTab: String, CaseIterable, Identifiable {
    case all = "全部"
    case apple = "Apple 内置"
    case local = "本地离线"
    case cloud = "云端与局域网"

    var id: String { self.rawValue }

    var icon: String {
        switch self {
        case .all: return "square.grid.2x2"
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

    @State private var selectedTab: ModelCategoryTab = .all
    @State private var isTestingConnection: Bool = false
    @State private var connectionTestMessage: String? = nil
    @State private var connectionTestSuccess: Bool? = nil

    private var activeModel: SpeechModel {
        self.settings.selectedSpeechModel
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            // 1. Header
            self.headerSection

            // 2. Active Model Status Card
            self.activeStatusCard

            // 3. Category Filter Tabs
            self.categoryTabPicker

            // 4. Model List
            self.modelListSection

            // 5. Detailed Configuration for Selected Engine (Ollama / Cloud / Custom)
            if self.activeModel.isCloudModel {
                self.cloudConfigurationSection
            }

            // 6. Intelligent Fallback & Timeout Settings
            self.smartFallbackSection

            // 7. Tip Section
            self.advancedOptionsSection
        }
        .padding(.horizontal, 4)
        .padding(.vertical, 8)
        .onAppear {
            self.viewModel.onAppear()
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

    // MARK: - 2. Active Status Card
    private var activeStatusCard: some View {
        HStack(spacing: 14) {
            ZStack {
                Circle()
                    .fill(Color.fluidGreen.opacity(0.15))
                    .frame(width: 44, height: 44)
                Image(systemName: self.iconForModel(self.activeModel))
                    .font(.system(size: 20))
                    .foregroundStyle(Color.fluidGreen)
            }

            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 6) {
                    Text(self.displayNameForModel(self.activeModel))
                        .font(.headline)
                        .fontWeight(.semibold)

                    Text("Active".loc)
                        .font(.system(size: 11, weight: .bold))
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

            if self.settings.cloudSTTAutoFallback && self.activeModel.isCloudModel {
                HStack(spacing: 4) {
                    Image(systemName: "shield.checkmark.fill")
                        .foregroundStyle(self.theme.palette.accent)
                    Text("Auto Fallback Enabled".loc)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .background(RoundedRectangle(cornerRadius: 6).fill(Color.secondary.opacity(0.1)))
            }
        }
        .padding(14)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color(nsColor: .controlBackgroundColor))
                .overlay(
                    RoundedRectangle(cornerRadius: 12)
                        .stroke(Color.secondary.opacity(0.2), lineWidth: 1)
                )
        )
    }

    // MARK: - 3. Category Tab Picker
    private var categoryTabPicker: some View {
        Picker("", selection: self.$selectedTab) {
            ForEach(ModelCategoryTab.allCases) { tab in
                Label(tab.rawValue.loc, systemImage: tab.icon).tag(tab)
            }
        }
        .pickerStyle(.segmented)
        .padding(.vertical, 2)
    }

    // MARK: - 4. Model List Section
    private var modelListSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(self.categorySectionTitle)
                .font(.headline)
                .foregroundStyle(.primary)

            VStack(spacing: 8) {
                ForEach(self.displayedModels) { model in
                    self.modelRow(for: model)
                }
            }
        }
    }

    private var categorySectionTitle: String {
        switch self.selectedTab {
        case .all: return "All Available Models".loc
        case .apple: return "Apple Built-in Engines (Zero Download)".loc
        case .local: return "On-Device Offline Models (Whisper & Neural)".loc
        case .cloud: return "Cloud & LAN Servers (Ollama / Cloud APIs)".loc
        }
    }

    private var displayedModels: [SpeechModel] {
        switch self.selectedTab {
        case .all:
            return SpeechModel.availableModels
        case .apple:
            return [.appleSpeech, .appleSpeechAnalyzer]
        case .local:
            return SpeechModel.availableModels.filter { !$0.isCloudModel && $0 != .appleSpeech && $0 != .appleSpeechAnalyzer }
        case .cloud:
            return SpeechModel.availableModels.filter { $0.isCloudModel }
        }
    }

    // MARK: - Model Card Row
    private func modelRow(for model: SpeechModel) -> some View {
        let isSelected = self.activeModel == model
        let isInstalled = model.isInstalled
        let isDownloading = self.viewModel.downloadingModel == model

        return VStack(spacing: 0) {
            HStack(spacing: 12) {
                // Icon
                Image(systemName: self.iconForModel(model))
                    .font(.system(size: 18))
                    .foregroundStyle(isSelected ? self.theme.palette.accent : .secondary)
                    .frame(width: 28, height: 28)

                // Info
                VStack(alignment: .leading, spacing: 2) {
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

                        if model == .whisperLargeTurbo || model == .appleSpeech {
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
        }
        .background(
            RoundedRectangle(cornerRadius: 10)
                .fill(isSelected ? self.theme.palette.accent.opacity(0.08) : Color(nsColor: .controlBackgroundColor).opacity(0.6))
                .overlay(
                    RoundedRectangle(cornerRadius: 10)
                        .stroke(isSelected ? self.theme.palette.accent.opacity(0.5) : Color.secondary.opacity(0.15), lineWidth: 1)
                )
        )
        .contentShape(Rectangle())
        .onTapGesture {
            if isInstalled && !isSelected {
                withAnimation(.easeInOut(duration: 0.15)) {
                    self.settings.selectedSpeechModel = model
                }
            }
        }
    }

    @ViewBuilder
    private func modelActionView(for model: SpeechModel, isSelected: Bool, isInstalled: Bool, isDownloading: Bool) -> some View {
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
            Button {
                withAnimation(.easeInOut(duration: 0.15)) {
                    self.settings.selectedSpeechModel = model
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

    private func miniBadge(text: String, color: Color) -> some View {
        Text(text)
            .font(.system(size: 9, weight: .bold))
            .padding(.horizontal, 5)
            .padding(.vertical, 1.5)
            .background(Capsule().fill(color.opacity(0.15)))
            .foregroundStyle(color)
    }

    // MARK: - 5. Cloud Configuration Section
    private var cloudConfigurationSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Engine Configuration".loc)
                .font(.headline)
                .foregroundStyle(.primary)

            VStack(spacing: 12) {
                // Base URL & Model for specific cloud engines
                if self.activeModel == .cloudOllama {
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
                } else if self.activeModel == .cloudOpenAI {
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
                } else if self.activeModel == .cloudGroq {
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
                } else if self.activeModel == .cloudOpenRouter {
                    HStack(alignment: .center) {
                        Text("OpenRouter API Key".loc)
                            .font(.subheadline)
                            .frame(width: 140, alignment: .leading)
                        SecureField("sk-or-...", text: self.$settings.cloudSTTOpenRouterAPIKey)
                            .textFieldStyle(.roundedBorder)
                    }
                } else if self.activeModel == .cloudCustom {
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

                // Target Language & Test Button
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
                    .frame(width: 160)

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

                // Optional ASR Prompt
                VStack(alignment: .leading, spacing: 4) {
                    Text("ASR Prompt (Optional / Filter Filler Words)".loc)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    TextField("e.g. Accurately transcribe audio, filter out verbal fillers like uh, um, etc.".loc, text: self.$settings.cloudSTTPrompt)
                        .textFieldStyle(.roundedBorder)
                        .font(.subheadline)
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

    // MARK: - 6. Smart Fallback Section
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

    // MARK: - 7. Advanced Options Section
    private var advancedOptionsSection: some View {
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
        case .parakeetTDT, .parakeetTDTv2, .parakeetRealtime: return "bird.fill"
        default: return "cpu.fill"
        }
    }

    private func displayNameForModel(_ model: SpeechModel) -> String {
        switch model {
        case .appleSpeech: return "Apple Speech (经典内置引擎)".loc
        case .appleSpeechAnalyzer: return "Apple Speech Analyzer (macOS 15+ 现代流式)".loc
        case .cloudOllama: return "Ollama ASR (局域网 / 本地服务器)".loc
        case .cloudOpenAI: return "OpenAI Cloud STT (Whisper API)".loc
        case .cloudGroq: return "Groq Cloud STT (极速 LPUs)".loc
        case .cloudOpenRouter: return "OpenRouter Cloud STT".loc
        case .cloudCustom: return "Custom Cloud STT (自定义接口)".loc
        case .parakeetTDT: return "Parakeet TDT v3 (中英多语言极速)".loc
        case .parakeetTDTv2: return "Parakeet TDT v2 (纯英文高精度)".loc
        case .parakeetRealtime: return "Parakeet Flash (实时流式)".loc
        case .qwen3Asr: return "Qwen3 ASR (千问本地离线)".loc
        case .cohereTranscribeSixBit: return "Cohere Transcribe (6-bit)".loc
        case .nemotronOffline: return "Nemotron 3.5 (NVIDIA 离线)".loc
        case .nemotronStreaming, .nemotronStreaming320: return "Nemotron Speech (低延迟流式)".loc
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
            return "macOS 15+ Sequoia 采用的新一代神经流式识别引擎，延迟更低、分词更精准。".loc
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
        case .parakeetTDT:
            return "FluidAudio 驱动的端到端超快多语言模型（支持 25 种语言），推理极速。".loc
        case .parakeetTDTv2:
            return "针对纯英文深度优化的端到端极速模型，极高准确率。".loc
        case .parakeetRealtime:
            return "支持边说边出字的原生流式离线模型。".loc
        case .qwen3Asr:
            return "阿里通义千问 Qwen3-ASR 本地离线神经网络模型，对中文理解极深。".loc
        case .cohereTranscribeSixBit:
            return "Cohere 6-bit 量化高精度离线模型。".loc
        case .nemotronOffline:
            return "NVIDIA Nemotron 3.5 离线多语言高精度语音模型。".loc
        case .nemotronStreaming, .nemotronStreaming320:
            return "NVIDIA Nemotron 超低延迟流式语音模型。".loc
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
        case .appleSpeechAnalyzer: return "macOS 15+ 现代神经流式引擎 (SpeechAnalyzer)".loc
        case .cloudOllama: return "\(self.settings.cloudSTTOllamaBaseURL) (\(self.settings.cloudSTTOllamaModel))"
        case .cloudOpenAI: return "OpenAI Cloud STT (\(self.settings.cloudSTTOpenAIModel))"
        case .cloudGroq: return "Groq Cloud STT (\(self.settings.cloudSTTGroqModel))"
        default: return "本地离线神经网络引擎 (\(model.downloadSize))".loc
        }
    }

    private func testConnection() {
        self.isTestingConnection = true
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
}
