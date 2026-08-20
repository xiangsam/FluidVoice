//
//  CloudSTTConfigView.swift
//  FluidVoice
//

import SwiftUI

struct CloudSTTConfigView: View {
    let model: SettingsStore.SpeechModel
    let theme: AppTheme
    @ObservedObject var settings: SettingsStore

    @State private var isShowingAPIKey = false
    @State private var isTesting = false
    @State private var testResultMessage: String?
    @State private var testResultSuccess: Bool?

    private var cloudSTTType: CloudSTTType {
        switch self.model {
        case .cloudOpenRouter: return .openRouter
        case .cloudOpenAI: return .openAI
        case .cloudGroq: return .groq
        case .cloudCustom: return .custom
        default: return .openRouter
        }
    }

    private var apiKeyBinding: Binding<String> {
        Binding(
            get: {
                switch self.cloudSTTType {
                case .openRouter: return self.settings.cloudSTTOpenRouterAPIKey
                case .openAI: return self.settings.cloudSTTOpenAIAPIKey
                case .groq: return self.settings.cloudSTTGroqAPIKey
                case .custom: return self.settings.cloudSTTCustomAPIKey
                }
            },
            set: { newValue in
                switch self.cloudSTTType {
                case .openRouter: self.settings.cloudSTTOpenRouterAPIKey = newValue
                case .openAI: self.settings.cloudSTTOpenAIAPIKey = newValue
                case .groq: self.settings.cloudSTTGroqAPIKey = newValue
                case .custom: self.settings.cloudSTTCustomAPIKey = newValue
                }
            }
        )
    }

    private var baseURLBinding: Binding<String> {
        Binding(
            get: {
                switch self.cloudSTTType {
                case .openRouter: return self.settings.cloudSTTOpenRouterBaseURL
                case .openAI: return self.settings.cloudSTTOpenAIBaseURL
                case .groq: return self.settings.cloudSTTGroqBaseURL
                case .custom: return self.settings.cloudSTTCustomBaseURL
                }
            },
            set: { newValue in
                switch self.cloudSTTType {
                case .openRouter: self.settings.cloudSTTOpenRouterBaseURL = newValue
                case .openAI: self.settings.cloudSTTOpenAIBaseURL = newValue
                case .groq: self.settings.cloudSTTGroqBaseURL = newValue
                case .custom: self.settings.cloudSTTCustomBaseURL = newValue
                }
            }
        )
    }

    private var modelBinding: Binding<String> {
        Binding(
            get: {
                switch self.cloudSTTType {
                case .openRouter: return self.settings.cloudSTTOpenRouterModel
                case .openAI: return self.settings.cloudSTTOpenAIModel
                case .groq: return self.settings.cloudSTTGroqModel
                case .custom: return self.settings.cloudSTTCustomModel
                }
            },
            set: { newValue in
                switch self.cloudSTTType {
                case .openRouter: self.settings.cloudSTTOpenRouterModel = newValue
                case .openAI: self.settings.cloudSTTOpenAIModel = newValue
                case .groq: self.settings.cloudSTTGroqModel = newValue
                case .custom: self.settings.cloudSTTCustomModel = newValue
                }
            }
        )
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 8) {
                Image(systemName: "cloud.fill")
                    .foregroundStyle(self.theme.palette.accent)
                Text("\("Configure".loc) \(self.model.displayName)")
                    .font(self.theme.typography.bodyStrong)
                    .foregroundStyle(.primary)
                Spacer()

                if let website = ModelRepository.shared.providerWebsiteURL(for: self.cloudSTTType.rawValue) {
                    Button {
                        if let url = URL(string: website.url) {
                            NSWorkspace.shared.open(url)
                        }
                    } label: {
                        HStack(spacing: 4) {
                            Text(website.label.loc)
                            Image(systemName: "arrow.up.right.square")
                        }
                        .font(self.theme.typography.bodySmall)
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(self.theme.palette.accent)
                }
            }

            // API Key Row
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Text("API Key".loc)
                        .font(self.theme.typography.bodySmallStrong)
                        .foregroundStyle(.secondary)
                    Spacer()

                    // Import from AI Enhancement
                    let aiEnhanceKey = self.settings.getAPIKey(for: self.cloudSTTType.rawValue) ?? ""
                    if !aiEnhanceKey.isEmpty, self.apiKeyBinding.wrappedValue.isEmpty {
                        Button("Use AI Enhancement API Key".loc) {
                            self.apiKeyBinding.wrappedValue = aiEnhanceKey
                        }
                        .buttonStyle(.plain)
                        .font(self.theme.typography.bodySmall)
                        .foregroundStyle(Color.fluidGreen)
                    }
                }

                HStack(spacing: 8) {
                    if self.isShowingAPIKey {
                        TextField("Enter your API Key".loc, text: self.apiKeyBinding)
                            .textFieldStyle(.roundedBorder)
                    } else {
                        SecureField("Enter your API Key".loc, text: self.apiKeyBinding)
                            .textFieldStyle(.roundedBorder)
                    }

                    Button {
                        self.isShowingAPIKey.toggle()
                    } label: {
                        Image(systemName: self.isShowingAPIKey ? "eye.slash" : "eye")
                            .foregroundStyle(.secondary)
                    }
                    .buttonStyle(.plain)
                    .help(self.isShowingAPIKey ? "Hide API Key".loc : "Show API Key".loc)
                }
            }

            // Model ID Row
            VStack(alignment: .leading, spacing: 6) {
                Text("Model Name / ID".loc)
                    .font(self.theme.typography.bodySmallStrong)
                    .foregroundStyle(.secondary)

                HStack(spacing: 8) {
                    TextField("e.g. openai/whisper-large-v3", text: self.modelBinding)
                        .textFieldStyle(.roundedBorder)

                    // Preset menu
                    Menu {
                        ForEach(self.cloudSTTType.recommendedModels, id: \.self) { preset in
                            Button(preset) {
                                self.modelBinding.wrappedValue = preset
                            }
                        }
                    } label: {
                        Text("Presets".loc)
                            .font(self.theme.typography.bodySmall)
                    }
                    .menuStyle(.borderedButton)
                    .fixedSize()
                }
            }

            // Base URL Row
            VStack(alignment: .leading, spacing: 6) {
                Text("Base URL".loc)
                    .font(self.theme.typography.bodySmallStrong)
                    .foregroundStyle(.secondary)

                HStack(spacing: 8) {
                    TextField(self.cloudSTTType.defaultBaseURL, text: self.baseURLBinding)
                        .textFieldStyle(.roundedBorder)

                    if self.baseURLBinding.wrappedValue != self.cloudSTTType.defaultBaseURL && !self.cloudSTTType.defaultBaseURL.isEmpty {
                        Button("Reset to Default".loc) {
                            self.baseURLBinding.wrappedValue = self.cloudSTTType.defaultBaseURL
                        }
                        .buttonStyle(.plain)
                        .font(self.theme.typography.bodySmall)
                        .foregroundStyle(self.theme.palette.accent)
                    }
                }
            }

            // Language Selection & Test Connection
            HStack(alignment: .center, spacing: 12) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Target Language".loc)
                        .font(self.theme.typography.bodySmallStrong)
                        .foregroundStyle(.secondary)

                    Picker("", selection: self.$settings.cloudSTTLanguage) {
                        Text("Auto Detect".loc).tag("auto")
                        Text("Chinese (Simplified)".loc).tag("zh")
                        Text("English".loc).tag("en")
                        Text("Japanese").tag("ja")
                        Text("Korean").tag("ko")
                        Text("French").tag("fr")
                        Text("German").tag("de")
                        Text("Spanish").tag("es")
                    }
                    .labelsHidden()
                    .frame(width: 160)
                }

                Spacer()

                VStack(alignment: .trailing, spacing: 4) {
                    Button {
                        self.testCloudConnection()
                    } label: {
                        HStack(spacing: 6) {
                            if self.isTesting {
                                ProgressView()
                                    .controlSize(.small)
                            } else {
                                Image(systemName: "network")
                            }
                            Text(self.isTesting ? "Testing...".loc : "Test Connection".loc)
                        }
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    .disabled(self.isTesting)

                    if let message = self.testResultMessage {
                        Text(message)
                            .font(self.theme.typography.bodySmall)
                            .foregroundStyle(self.testResultSuccess == true ? Color.fluidGreen : .red)
                            .lineLimit(2)
                    }
                }
            }
        }
        .padding(12)
        .background(
            RoundedRectangle(cornerRadius: 10)
                .fill(self.theme.palette.contentBackground.opacity(0.8))
                .overlay(
                    RoundedRectangle(cornerRadius: 10)
                        .stroke(self.theme.palette.cardBorder.opacity(0.4), lineWidth: 1)
                )
        )
    }

    private func testCloudConnection() {
        self.isTesting = true
        self.testResultMessage = nil
        self.testResultSuccess = nil

        let provider = CloudTranscriptionProvider(type: self.cloudSTTType, settings: self.settings)

        Task {
            do {
                // Test with 0.5s of audio samples (PCM 16kHz silence)
                let dummySamples: [Float] = Array(repeating: 0.0, count: 8000)
                let _ = try await provider.transcribe(dummySamples)
                await MainActor.run {
                    self.isTesting = false
                    self.testResultSuccess = true
                    self.testResultMessage = "Connection successful!".loc
                }
            } catch {
                await MainActor.run {
                    self.isTesting = false
                    self.testResultSuccess = false
                    self.testResultMessage = "\("Connection failed".loc): \(error.localizedDescription)"
                }
            }
        }
    }
}
