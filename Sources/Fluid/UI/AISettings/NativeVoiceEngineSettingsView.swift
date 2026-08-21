//
//  NativeVoiceEngineSettingsView.swift
//  FluidVoice
//
//  Modern, native SwiftUI Voice Engine Settings View adhering to Apple HIG.
//

import SwiftUI

typealias SpeechModel = SettingsStore.SpeechModel

struct NativeVoiceEngineSettingsView: View {
    @ObservedObject var viewModel: VoiceEngineSettingsViewModel
    @ObservedObject var settings: SettingsStore
    let theme: AppTheme

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

            // 3. Engine Category / Selection
            self.engineSelectionSection

            // 4. Detailed Configuration for Selected Engine (Ollama / Cloud / Local)
            if self.activeModel.isCloudModel || self.activeModel == .cloudOllama {
                self.cloudConfigurationSection
            }

            // 5. Intelligent Fallback & Timeout Settings
            self.smartFallbackSection

            // 6. Filler Words & Formatting Rules Quick Link
            self.advancedOptionsSection
        }
        .padding(.horizontal, 4)
        .padding(.vertical, 8)
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
                    Text(self.activeModel.displayName)
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

            if self.settings.cloudSTTAutoFallback && (self.activeModel.isCloudModel || self.activeModel == .cloudOllama) {
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

    // MARK: - 3. Engine Selection Section
    private var engineSelectionSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Choose Engine".loc)
                .font(.headline)
                .foregroundStyle(.primary)

            VStack(spacing: 8) {
                // Option A: macOS Native Apple Speech
                self.engineOptionRow(
                    model: .appleSpeech,
                    icon: "apple.logo",
                    title: "Apple Speech (macOS Native)".loc,
                    subtitle: "Zero latency, fully offline, privacy-first, built into macOS.".loc,
                    badge: "Recommended".loc
                )

                // Option B: Ollama ASR (Local / LAN)
                self.engineOptionRow(
                    model: .cloudOllama,
                    icon: "network",
                    title: "Ollama ASR (Local / LAN Server)".loc,
                    subtitle: "Connects to your local or remote Ollama server (e.g. Qwen3-ASR, SenseVoice).".loc,
                    badge: "Custom".loc
                )

                // Option C: OpenAI Cloud ASR
                self.engineOptionRow(
                    model: .cloudOpenAI,
                    icon: "cloud.fill",
                    title: "OpenAI Cloud STT (Whisper API)".loc,
                    subtitle: "Official high-accuracy cloud transcription via OpenAI API.".loc
                )

                // Option D: Groq Cloud ASR
                self.engineOptionRow(
                    model: .cloudGroq,
                    icon: "bolt.fill",
                    title: "Groq Cloud STT (Ultra Fast)".loc,
                    subtitle: "Near-instant cloud transcription powered by Groq LPUs.".loc
                )

                // Option E: Local Whisper
                self.engineOptionRow(
                    model: .whisperLargeTurbo,
                    icon: "cpu.fill",
                    title: "Whisper Large Turbo (Local Offline)".loc,
                    subtitle: "High precision offline model running locally on your Mac.".loc
                )
            }
        }
    }

    private func engineOptionRow(model: SpeechModel, icon: String, title: String, subtitle: String, badge: String? = nil) -> some View {
        let isSelected = self.activeModel == model

        return Button {
            withAnimation(.easeInOut(duration: 0.15)) {
                self.settings.selectedSpeechModel = model
            }
        } label: {
            HStack(spacing: 12) {
                Image(systemName: icon)
                    .font(.system(size: 18))
                    .foregroundStyle(isSelected ? self.theme.palette.accent : .secondary)
                    .frame(width: 28, height: 28)

                VStack(alignment: .leading, spacing: 2) {
                    HStack(spacing: 6) {
                        Text(title)
                            .font(.system(size: 13, weight: isSelected ? .semibold : .regular))
                            .foregroundStyle(.primary)

                        if let badge {
                            Text(badge)
                                .font(.system(size: 10, weight: .bold))
                                .padding(.horizontal, 5)
                                .padding(.vertical, 1)
                                .background(Capsule().fill(self.theme.palette.accent.opacity(0.15)))
                                .foregroundStyle(self.theme.palette.accent)
                        }
                    }

                    Text(subtitle)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }

                Spacer()

                if isSelected {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundStyle(self.theme.palette.accent)
                        .font(.system(size: 16))
                } else {
                    Circle()
                        .strokeBorder(Color.secondary.opacity(0.3), lineWidth: 1)
                        .frame(width: 16, height: 16)
                }
            }
            .padding(12)
            .background(
                RoundedRectangle(cornerRadius: 10)
                    .fill(isSelected ? self.theme.palette.accent.opacity(0.08) : Color(nsColor: .controlBackgroundColor).opacity(0.6))
                    .overlay(
                        RoundedRectangle(cornerRadius: 10)
                            .stroke(isSelected ? self.theme.palette.accent.opacity(0.5) : Color.secondary.opacity(0.15), lineWidth: 1)
                    )
            )
        }
        .buttonStyle(.plain)
    }

    // MARK: - 4. Cloud Configuration Section
    private var cloudConfigurationSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Engine Configuration".loc)
                .font(.headline)
                .foregroundStyle(.primary)

            VStack(spacing: 12) {
                // Base URL
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

    // MARK: - 6. Advanced Options Section
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

    // MARK: - Helpers
    private func iconForModel(_ model: SpeechModel) -> String {
        switch model {
        case .appleSpeech, .appleSpeechAnalyzer: return "apple.logo"
        case .cloudOllama: return "network"
        case .cloudOpenAI, .cloudGroq, .cloudOpenRouter, .cloudCustom: return "cloud.fill"
        default: return "cpu.fill"
        }
    }

    private func subtitleForModel(_ model: SpeechModel) -> String {
        switch model {
        case .appleSpeech: return "macOS Built-in Speech Recognition Engine".loc
        case .cloudOllama: return "\(self.settings.cloudSTTOllamaBaseURL) (\(self.settings.cloudSTTOllamaModel))"
        case .cloudOpenAI: return "OpenAI Cloud STT (\(self.settings.cloudSTTOpenAIModel))"
        case .cloudGroq: return "Groq Cloud STT (\(self.settings.cloudSTTGroqModel))"
        default: return "Local Neural Engine Model".loc
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
