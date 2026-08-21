//
//  CloudSTTConfigView.swift
//  FluidVoice
//

import SwiftUI

struct CloudSTTConfigView: View {
    let model: SettingsStore.SpeechModel
    let theme: AppTheme
    @ObservedObject var settings: SettingsStore
    var onClose: (() -> Void)? = nil

    @State private var isShowingAPIKey = false
    @State private var isTesting = false
    @State private var testResultMessage: String?
    @State private var testResultSuccess: Bool?
    @State private var searchText = ""
    @State private var showCustomModelInput = false
    @State private var copiedModelID: String?
    @State private var fetchedOllamaModels: [String] = []
    @State private var isFetchingOllamaModels = false

    private var cloudSTTType: CloudSTTType {
        switch self.model {
        case .cloudOpenRouter: return .openRouter
        case .cloudOpenAI: return .openAI
        case .cloudGroq: return .groq
        case .cloudOllama: return .ollama
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
                case .ollama: return self.settings.cloudSTTOllamaAPIKey
                case .custom: return self.settings.cloudSTTCustomAPIKey
                }
            },
            set: { newValue in
                switch self.cloudSTTType {
                case .openRouter: self.settings.cloudSTTOpenRouterAPIKey = newValue
                case .openAI: self.settings.cloudSTTOpenAIAPIKey = newValue
                case .groq: self.settings.cloudSTTGroqAPIKey = newValue
                case .ollama: self.settings.cloudSTTOllamaAPIKey = newValue
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
                case .ollama: return self.settings.cloudSTTOllamaBaseURL
                case .custom: return self.settings.cloudSTTCustomBaseURL
                }
            },
            set: { newValue in
                switch self.cloudSTTType {
                case .openRouter: self.settings.cloudSTTOpenRouterBaseURL = newValue
                case .openAI: self.settings.cloudSTTOpenAIBaseURL = newValue
                case .groq: self.settings.cloudSTTGroqBaseURL = newValue
                case .ollama: self.settings.cloudSTTOllamaBaseURL = newValue
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
                case .ollama: return self.settings.cloudSTTOllamaModel
                case .custom: return self.settings.cloudSTTCustomModel
                }
            },
            set: { newValue in
                switch self.cloudSTTType {
                case .openRouter: self.settings.cloudSTTOpenRouterModel = newValue
                case .openAI: self.settings.cloudSTTOpenAIModel = newValue
                case .groq: self.settings.cloudSTTGroqModel = newValue
                case .ollama: self.settings.cloudSTTOllamaModel = newValue
                case .custom: self.settings.cloudSTTCustomModel = newValue
                }
            }
        )
    }

    private var filteredModels: [CloudSTTModelItem] {
        var presets = self.cloudSTTType.modelPresets
        if self.cloudSTTType == .ollama && !self.fetchedOllamaModels.isEmpty {
            let dynamicItems = self.fetchedOllamaModels.map { name in
                CloudSTTModelItem(
                    id: name,
                    name: name,
                    vendor: "Ollama Node",
                    tag: name.contains("qwen") ? "🎯 ASR" : "Node Model",
                    priceHint: "Local / Free",
                    isPopular: true
                )
            }
            presets = dynamicItems
        }

        if self.searchText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return presets
        }
        let query = self.searchText.lowercased()
        return presets.filter {
            $0.name.lowercased().contains(query)
                || $0.id.lowercased().contains(query)
                || $0.vendor.lowercased().contains(query)
                || ($0.tag?.lowercased().contains(query) ?? false)
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            // Header
            HStack(spacing: 8) {
                Image(systemName: "cloud.fill")
                    .foregroundStyle(self.theme.palette.accent)
                Text("\("Configure".loc) \(self.model.displayName)")
                    .font(self.theme.typography.bodyStrong)
                    .foregroundStyle(.primary)
                Spacer()

                if self.cloudSTTType == .ollama {
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
                        .font(self.theme.typography.bodySmall)
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                }

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

                if let onClose = self.onClose {
                    Button {
                        onClose()
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .font(.system(size: 16))
                            .foregroundStyle(.secondary)
                    }
                    .buttonStyle(.plain)
                    .help("Close Configuration".loc)
                }
            }

            // API Key Row
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Text("API Key".loc)
                        .font(self.theme.typography.bodySmallStrong)
                        .foregroundStyle(.secondary)
                    Spacer()

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

            Divider().opacity(0.3)

            // Embedded Model Selection Area
            VStack(alignment: .leading, spacing: 10) {
                HStack(alignment: .center) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Speech-to-Text Model".loc)
                            .font(self.theme.typography.bodyStrong)
                            .foregroundStyle(.primary)

                        let currentModelID = self.modelBinding.wrappedValue.isEmpty ? self.cloudSTTType.defaultModel : self.modelBinding.wrappedValue
                        Text("\("Active Selection:".loc) \(currentModelID)")
                            .font(self.theme.typography.caption)
                            .foregroundStyle(self.theme.palette.accent)
                    }

                    Spacer()

                    // Search input
                    HStack(spacing: 6) {
                        Image(systemName: "magnifyingglass")
                            .font(.system(size: 11))
                            .foregroundStyle(.secondary)
                        TextField("Search models...".loc, text: self.$searchText)
                            .textFieldStyle(.plain)
                            .font(self.theme.typography.bodySmall)
                            .frame(width: 130)

                        if !self.searchText.isEmpty {
                            Button {
                                self.searchText = ""
                            } label: {
                                Image(systemName: "xmark.circle.fill")
                                    .font(.system(size: 11))
                                    .foregroundStyle(.secondary)
                            }
                            .buttonStyle(.plain)
                        }
                    }
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(
                        RoundedRectangle(cornerRadius: 7)
                            .fill(self.theme.palette.contentBackground)
                            .overlay(
                                RoundedRectangle(cornerRadius: 7)
                                    .stroke(self.theme.palette.cardBorder.opacity(0.5), lineWidth: 1)
                            )
                    )
                }

                // Scrollable Embedded Models Card Grid/List
                ScrollView(.vertical, showsIndicators: true) {
                    LazyVStack(spacing: 6) {
                        ForEach(self.filteredModels) { item in
                            self.modelCardRow(item: item)
                        }

                        if self.filteredModels.isEmpty {
                            VStack(spacing: 6) {
                                Image(systemName: "magnifyingglass")
                                    .font(.title3)
                                    .foregroundStyle(.secondary)
                                Text("No matching models found".loc)
                                    .font(self.theme.typography.bodySmall)
                                    .foregroundStyle(.secondary)
                            }
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 24)
                        }
                    }
                    .padding(6)
                }
                .frame(height: 230)
                .background(
                    RoundedRectangle(cornerRadius: 10)
                        .fill(self.theme.palette.contentBackground.opacity(0.6))
                        .overlay(
                            RoundedRectangle(cornerRadius: 10)
                                .stroke(self.theme.palette.cardBorder.opacity(0.4), lineWidth: 1)
                        )
                )
                .clipped()

                // Custom Model ID Input Toggle
                VStack(alignment: .leading, spacing: 6) {
                    Button {
                        withAnimation(.easeInOut(duration: 0.2)) {
                            self.showCustomModelInput.toggle()
                        }
                    } label: {
                        HStack(spacing: 4) {
                            Image(systemName: self.showCustomModelInput ? "chevron.down" : "chevron.right")
                                .font(.system(size: 10))
                            Text("Custom Model ID".loc)
                                .font(self.theme.typography.bodySmall)
                        }
                        .foregroundStyle(self.theme.palette.accent)
                    }
                    .buttonStyle(.plain)

                    if self.showCustomModelInput {
                        HStack(spacing: 8) {
                            TextField("Enter any custom model ID, e.g. openai/whisper-large-v3", text: self.modelBinding)
                                .textFieldStyle(.roundedBorder)
                                .font(self.theme.typography.bodySmall)

                            if !self.modelBinding.wrappedValue.isEmpty {
                                Button("Reset".loc) {
                                    self.modelBinding.wrappedValue = self.cloudSTTType.defaultModel
                                }
                                .buttonStyle(.bordered)
                                .controlSize(.small)
                            }
                        }
                    }
                }
            }

            Divider().opacity(0.3)

            // Target Language & Connection Test
            HStack(alignment: .center, spacing: 16) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Target Language".loc)
                        .font(self.theme.typography.bodySmallStrong)
                        .foregroundStyle(.secondary)

                    Picker("", selection: self.$settings.cloudSTTLanguage) {
                        Text("Auto Detect".loc).tag("auto")
                        Text("Chinese (Simplified)".loc).tag("zh")
                        Text("English".loc).tag("en")
                        Text("Japanese".loc).tag("ja")
                        Text("Korean".loc).tag("ko")
                        Text("French".loc).tag("fr")
                        Text("German".loc).tag("de")
                        Text("Spanish".loc).tag("es")
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

            VStack(alignment: .leading, spacing: 4) {
                Text("ASR Prompt (Optional / Filter Filler Words)".loc)
                    .font(self.theme.typography.bodySmallStrong)
                    .foregroundStyle(.secondary)

                TextField("e.g. Accurately transcribe audio, filter out verbal fillers like uh, um, etc.".loc, text: self.$settings.cloudSTTPrompt)
                    .textFieldStyle(.roundedBorder)
                    .font(.subheadline)
            }

            Divider().opacity(0.4)

            // Intelligent Fallback & Timeout Controls
            VStack(alignment: .leading, spacing: 10) {
                HStack {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("macOS 本地语音智能兜底 (Apple Speech)".loc)
                            .font(self.theme.typography.bodyStrong)
                        Text("当云端或 Ollama 响应超时或发生异常时，自动无缝切换到系统内置本地引擎转录，保障录音 100% 不丢失。".loc)
                            .font(self.theme.typography.caption)
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                    Toggle("", isOn: self.$settings.cloudSTTAutoFallback)
                        .labelsHidden()
                        .toggleStyle(.switch)
                }

                HStack {
                    Text("响应超时时间".loc)
                        .font(self.theme.typography.bodySmallStrong)
                        .foregroundStyle(.secondary)
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

            // Bottom Actions: Done / Return
            if let onClose = self.onClose {
                Divider().opacity(0.3)

                HStack {
                    Spacer()
                    Button {
                        onClose()
                    } label: {
                        HStack(spacing: 6) {
                            Image(systemName: "checkmark.circle.fill")
                            Text("Done / Return to Models".loc)
                        }
                        .font(self.theme.typography.bodySmallStrong)
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(Color.fluidGreen)
                    .controlSize(.regular)
                }
            }
        }
        .padding(14)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(self.theme.palette.cardBackground)
                .overlay(
                    RoundedRectangle(cornerRadius: 12)
                        .stroke(self.theme.palette.cardBorder.opacity(0.3), lineWidth: 1)
                )
        )
        .onAppear {
            if self.cloudSTTType == .ollama {
                self.fetchOllamaModels()
            }
        }
    }

    // MARK: - Model Card Row Component

    private func modelCardRow(item: CloudSTTModelItem) -> some View {
        let currentModel = self.modelBinding.wrappedValue.isEmpty ? self.cloudSTTType.defaultModel : self.modelBinding.wrappedValue
        let isSelected = currentModel == item.id
        let vendorTheme = self.vendorInfo(for: item.vendor)

        return Button {
            self.modelBinding.wrappedValue = item.id
        } label: {
            HStack(spacing: 10) {
                // Vendor Icon Badge
                ZStack {
                    RoundedRectangle(cornerRadius: 6)
                        .fill(vendorTheme.color.opacity(0.18))
                        .frame(width: 28, height: 28)
                    Image(systemName: vendorTheme.icon)
                        .font(.system(size: 13, weight: .semibold))
                        .foregroundStyle(vendorTheme.color)
                }

                // Name & ID
                VStack(alignment: .leading, spacing: 2) {
                    HStack(spacing: 6) {
                        Text(item.name)
                            .font(self.theme.typography.bodyStrong)
                            .foregroundStyle(isSelected ? Color.primary : Color.primary.opacity(0.85))

                        if let tag = item.tag {
                            Text(tag.loc)
                                .font(.system(size: 9, weight: .semibold))
                                .padding(.horizontal, 5)
                                .padding(.vertical, 1.5)
                                .background(
                                    Capsule()
                                        .fill(item.isPopular ? Color.orange.opacity(0.2) : Color.blue.opacity(0.2))
                                )
                                .foregroundStyle(item.isPopular ? Color.orange : Color.blue)
                        }
                    }

                    HStack(spacing: 4) {
                        Text(item.id)
                            .font(.system(size: 11, design: .monospaced))
                            .foregroundStyle(.secondary)

                        Button {
                            NSPasteboard.general.clearContents()
                            NSPasteboard.general.setString(item.id, forType: .string)
                            self.copiedModelID = item.id
                            DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
                                if self.copiedModelID == item.id {
                                    self.copiedModelID = nil
                                }
                            }
                        } label: {
                            Image(systemName: self.copiedModelID == item.id ? "checkmark" : "doc.on.doc")
                                .font(.system(size: 10))
                                .foregroundStyle(self.copiedModelID == item.id ? Color.fluidGreen : .secondary.opacity(0.6))
                        }
                        .buttonStyle(.plain)
                        .help("Copy Model ID".loc)
                    }
                }

                Spacer()

                // Price Hint & Selection Checkmark
                HStack(spacing: 10) {
                    if let price = item.priceHint {
                        Text(price)
                            .font(.system(size: 11, design: .monospaced))
                            .foregroundStyle(.secondary)
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(
                                RoundedRectangle(cornerRadius: 4)
                                    .fill(self.theme.palette.cardBackground.opacity(0.8))
                            )
                    }

                    ZStack {
                        Circle()
                            .stroke(isSelected ? Color.fluidGreen : self.theme.palette.cardBorder.opacity(0.4), lineWidth: 1.5)
                            .frame(width: 18, height: 18)

                        if isSelected {
                            Circle()
                                .fill(Color.fluidGreen)
                                .frame(width: 10, height: 10)
                        }
                    }
                }
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 8)
            .background(
                RoundedRectangle(cornerRadius: 8)
                    .fill(isSelected ? Color.fluidGreen.opacity(0.1) : self.theme.palette.cardBackground.opacity(0.5))
                    .overlay(
                        RoundedRectangle(cornerRadius: 8)
                            .stroke(isSelected ? Color.fluidGreen.opacity(0.6) : Color.clear, lineWidth: 1.2)
                    )
            )
        }
        .buttonStyle(.plain)
    }

    private func vendorInfo(for vendor: String) -> (icon: String, color: Color) {
        let lower = vendor.lowercased()
        if lower.contains("openai") {
            return ("sparkle", Color(red: 0.06, green: 0.65, blue: 0.52))
        } else if lower.contains("mistral") {
            return ("flame.fill", Color(red: 0.98, green: 0.45, blue: 0.18))
        } else if lower.contains("nvidia") {
            return ("cpu", Color(red: 0.46, green: 0.72, blue: 0.0))
        } else if lower.contains("qwen") || lower.contains("alibaba") {
            return ("cube.transparent.fill", Color(red: 0.38, green: 0.38, blue: 0.92))
        } else if lower.contains("google") {
            return ("sparkles", Color(red: 0.26, green: 0.52, blue: 0.96))
        } else if lower.contains("xai") || lower.contains("spacex") {
            return ("bolt.horizontal.fill", Color.white.opacity(0.9))
        } else if lower.contains("deepgram") {
            return ("waveform.badge.magnifyingglass", Color(red: 0.12, green: 0.84, blue: 0.64))
        } else if lower.contains("microsoft") {
            return ("square.grid.2x2.fill", Color(red: 0.0, green: 0.64, blue: 0.94))
        } else if lower.contains("fish") {
            return ("waveform", Color(red: 0.96, green: 0.36, blue: 0.62))
        } else if lower.contains("groq") {
            return ("bolt.fill", Color.orange)
        } else {
            return ("cloud.fill", Color.blue)
        }
    }

    // MARK: - Connectivity Test

    private func testCloudConnection() {
        self.isTesting = true
        self.testResultMessage = nil
        self.testResultSuccess = nil

        Task {
            let provider = CloudTranscriptionProvider(type: self.cloudSTTType, settings: self.settings)
            do {
                let dummyAudio = [Float](repeating: 0.0, count: 16000)
                _ = try await provider.transcribe(dummyAudio)
                await MainActor.run {
                    self.isTesting = false
                    self.testResultSuccess = true
                    self.testResultMessage = "✓ " + ("Connection successful!".loc)
                }
            } catch {
                await MainActor.run {
                    self.isTesting = false
                    self.testResultSuccess = false
                    self.testResultMessage = "✕ " + error.localizedDescription
                }
            }
        }
    }

    private func fetchOllamaModels() {
        let rawBase = self.baseURLBinding.wrappedValue.trimmingCharacters(in: .whitespacesAndNewlines).trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        let base = rawBase.isEmpty ? "http://localhost:11434" : rawBase
        guard let url = URL(string: "\(base)/api/tags") else { return }

        self.isFetchingOllamaModels = true
        Task {
            defer {
                Task { @MainActor in
                    self.isFetchingOllamaModels = false
                }
            }
            do {
                let (data, _) = try await URLSession.shared.data(from: url)
                if let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                   let modelsArray = json["models"] as? [[String: Any]] {
                    let names = modelsArray.compactMap { $0["name"] as? String }
                    await MainActor.run {
                        self.fetchedOllamaModels = names
                    }
                }
            } catch {
                DebugLogger.shared.error("Failed to fetch Ollama models: \(error)", source: "CloudSTTConfigView")
            }
        }
    }
}
