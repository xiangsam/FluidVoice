//
//  NativeAIEnhancementSettingsView.swift
//  FluidVoice
//
//  Modern, native SwiftUI AI Enhancement & Post-Processing View adhering to Apple HIG.
//

import SwiftUI

struct NativeAIEnhancementSettingsView: View {
    @ObservedObject var viewModel: AIEnhancementSettingsViewModel
    @ObservedObject var settings: SettingsStore
    @ObservedObject var promptTest: DictationPromptTestCoordinator
    let theme: AppTheme

    @State private var isTestingConnection: Bool = false
    @State private var connectionMessage: String? = nil
    @State private var connectionSuccess: Bool? = nil

    @State private var sandboxInput: String = "呃，那个，我今天测试了一下Ollama模型，感觉速度还可以啊。"
    @State private var sandboxOutput: String = ""
    @State private var isTestingSandbox: Bool = false

    private var isAIEnabledBinding: Binding<Bool> {
        Binding(
            get: { self.settings.dictationPromptSelection != .off },
            set: { enabled in
                self.settings.setDictationPromptSelection(enabled ? .default : .off)
            }
        )
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            // 1. Header
            self.headerSection

            // 2. Master Toggle Card
            self.masterToggleCard

            if self.isAIEnabledBinding.wrappedValue {
                // 3. Provider & Model Configuration
                self.providerConfigurationSection

                // 4. Prompt Presets & Custom Instructions
                self.promptPresetsSection

                // 5. Interactive Testing Sandbox
                self.sandboxSection
            }
        }
        .padding(.horizontal, 4)
        .padding(.vertical, 8)
    }

    // MARK: - 1. Header
    private var headerSection: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 8) {
                Image(systemName: "sparkles.rectangle.stack.fill")
                    .font(.system(size: 24))
                    .foregroundStyle(self.theme.palette.accent)
                Text("AI Enhancements".loc)
                    .font(.title2)
                    .fontWeight(.bold)
            }
            Text("Automatically refine voice transcripts using LLMs for filler-word removal, grammar cleanup, and intelligent formatting.".loc)
                .font(.subheadline)
                .foregroundStyle(.secondary)
        }
    }

    // MARK: - 2. Master Toggle Card
    private var masterToggleCard: some View {
        HStack(spacing: 14) {
            ZStack {
                Circle()
                    .fill(self.isAIEnabledBinding.wrappedValue ? self.theme.palette.accent.opacity(0.15) : Color.secondary.opacity(0.15))
                    .frame(width: 44, height: 44)
                Image(systemName: "brain.head.profile")
                    .font(.system(size: 20))
                    .foregroundStyle(self.isAIEnabledBinding.wrappedValue ? self.theme.palette.accent : .secondary)
            }

            VStack(alignment: .leading, spacing: 3) {
                Text("Enable AI Post-Processing".loc)
                    .font(.headline)
                    .fontWeight(.semibold)

                Text(self.isAIEnabledBinding.wrappedValue ? "AI enhancements will run automatically after recording finishes.".loc : "Transcripts will be typed directly without LLM post-processing.".loc)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Spacer()

            Toggle("", isOn: self.isAIEnabledBinding)
                .labelsHidden()
                .toggleStyle(.switch)
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

    // MARK: - 3. Provider & Model Configuration
    private var providerConfigurationSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("AI Model Provider".loc)
                .font(.headline)
                .foregroundStyle(.primary)

            VStack(spacing: 12) {
                // Provider Selection
                HStack(alignment: .center) {
                    Text("Provider".loc)
                        .font(.subheadline)
                        .frame(width: 130, alignment: .leading)

                    Picker("", selection: self.$settings.selectedProviderID) {
                        Text("Ollama (Local / LAN)".loc).tag("ollama")
                        Text("OpenAI").tag("openai")
                        Text("DeepSeek").tag("deepseek")
                        Text("OpenRouter").tag("openrouter")
                        Text("Groq").tag("groq")
                        Text("Custom Provider".loc).tag("custom")
                    }
                    .labelsHidden()
                    .frame(maxWidth: .infinity, alignment: .leading)
                }

                // Base URL
                HStack(alignment: .center) {
                    Text("Base URL".loc)
                        .font(.subheadline)
                        .frame(width: 130, alignment: .leading)
                    TextField(self.defaultBaseURL(for: self.settings.selectedProviderID), text: self.baseURLBinding)
                        .textFieldStyle(.roundedBorder)
                }

                // API Key (for non-local providers)
                if self.settings.selectedProviderID != "ollama" {
                    HStack(alignment: .center) {
                        Text("API Key".loc)
                            .font(.subheadline)
                            .frame(width: 130, alignment: .leading)
                        SecureField("API Key...", text: self.apiKeyBinding)
                            .textFieldStyle(.roundedBorder)
                    }
                }

                // Model Name
                HStack(alignment: .center) {
                    Text("Model Name".loc)
                        .font(.subheadline)
                        .frame(width: 130, alignment: .leading)
                    TextField(self.defaultModel(for: self.settings.selectedProviderID), text: self.modelBinding)
                        .textFieldStyle(.roundedBorder)
                }

                // Test Connection Row
                HStack {
                    Spacer()

                    Button {
                        self.testLLMConnection()
                    } label: {
                        HStack(spacing: 4) {
                            if self.isTestingConnection {
                                ProgressView().controlSize(.small)
                            } else {
                                Image(systemName: "bolt.horizontal.fill")
                            }
                            Text(self.isTestingConnection ? "Testing...".loc : "Test Connection".loc)
                        }
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    .disabled(self.isTestingConnection)
                }

                if let message = self.connectionMessage {
                    HStack {
                        Spacer()
                        Text(message)
                            .font(.caption)
                            .foregroundStyle(self.connectionSuccess == true ? Color.fluidGreen : .red)
                    }
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

    // MARK: - 4. Prompt Presets Section
    private var promptPresetsSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Refinement Rules & Prompt".loc)
                .font(.headline)
                .foregroundStyle(.primary)

            VStack(alignment: .leading, spacing: 12) {
                // Quick Presets
                Text("Select Built-in Preset:".loc)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)

                HStack(spacing: 8) {
                    self.presetButton(
                        title: "🧹 去除语气词与杂音".loc,
                        prompt: "你是一个专业的语音文本整理助手。请准确保留用户的全部原意，自动去除语音中的'嗯'、'啊'、'那个'、'就是'、'然后'等口语语气词和重复词，修正错别字并输出规范流畅的中文。"
                    )
                    self.presetButton(
                        title: "✍️ 口语转标准书面语".loc,
                        prompt: "请将用户的口语表述整理为优雅、连贯的标准书面语，修复病句与逻辑停顿，适当补充规范标点符号，保持说话者的核心原意不变。"
                    )
                }

                HStack(spacing: 8) {
                    self.presetButton(
                        title: "🔤 中英文混排与标点".loc,
                        prompt: "请规范用户的文本排版：在中文字符与英文单词、数字之间自动添加标准空格，纠正漏标点和误标点，输出排版规范的中文。"
                    )
                    self.presetButton(
                        title: "💻 编程与代码模式".loc,
                        prompt: "用户正在口述编程与技术内容。请精准保留所有的技术术语、库名、函数名与变量名（如 camelCase、snake_case），输出清晰简洁的开发笔记格式。"
                    )
                }

                Divider().opacity(0.3)

                // Editable Prompt Text Box
                VStack(alignment: .leading, spacing: 4) {
                    Text("Active System Prompt:".loc)
                        .font(.subheadline)
                        .fontWeight(.medium)

                    TextEditor(text: self.activeSystemPromptBinding)
                        .font(.system(.subheadline, design: .monospaced))
                        .frame(minHeight: 90)
                        .padding(4)
                        .background(RoundedRectangle(cornerRadius: 8).fill(Color(nsColor: .textBackgroundColor)))
                        .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.secondary.opacity(0.2), lineWidth: 1))
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

    private func presetButton(title: String, prompt: String) -> some View {
        Button {
            withAnimation {
                self.settings.dictationCustomPromptText = prompt
                self.settings.setDictationPromptSelection(.default)
            }
        } label: {
            HStack {
                Text(title)
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(.primary)
                Spacer()
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 8)
            .background(
                RoundedRectangle(cornerRadius: 8)
                    .fill(Color.secondary.opacity(0.08))
                    .overlay(
                        RoundedRectangle(cornerRadius: 8)
                            .stroke(Color.secondary.opacity(0.15), lineWidth: 1)
                    )
            )
        }
        .buttonStyle(.plain)
    }

    // MARK: - 5. Interactive Testing Sandbox
    private var sandboxSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Interactive Sandbox Test".loc)
                .font(.headline)
                .foregroundStyle(.primary)

            VStack(alignment: .leading, spacing: 10) {
                Text("Test your current AI model & prompt with sample text:".loc)
                    .font(.caption)
                    .foregroundStyle(.secondary)

                HStack(alignment: .top, spacing: 10) {
                    // Input Column
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Input Text".loc)
                            .font(.caption)
                            .fontWeight(.medium)
                        TextField("Enter sample raw transcript...", text: self.$sandboxInput)
                            .textFieldStyle(.roundedBorder)
                    }

                    // Run Button
                    Button {
                        self.runSandboxTest()
                    } label: {
                        HStack(spacing: 4) {
                            if self.isTestingSandbox {
                                ProgressView().controlSize(.small)
                            } else {
                                Image(systemName: "play.fill")
                            }
                            Text(self.isTestingSandbox ? "Running...".loc : "Test AI".loc)
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.regular)
                    .disabled(self.isTestingSandbox || self.sandboxInput.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                    .padding(.top, 18)
                }

                if !self.sandboxOutput.isEmpty {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("AI Refined Result:".loc)
                            .font(.caption)
                            .fontWeight(.medium)
                            .foregroundStyle(Color.fluidGreen)

                        Text(self.sandboxOutput)
                            .font(.system(.subheadline, design: .default))
                            .padding(10)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .background(RoundedRectangle(cornerRadius: 8).fill(Color.fluidGreen.opacity(0.08)))
                            .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.fluidGreen.opacity(0.25), lineWidth: 1))
                    }
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

    // MARK: - Bindings & Helpers
    private var baseURLBinding: Binding<String> {
        Binding(
            get: {
                let pid = self.settings.selectedProviderID
                if let saved = self.settings.savedProviders.first(where: { $0.id == pid }), !saved.baseURL.isEmpty {
                    return saved.baseURL
                }
                return self.defaultBaseURL(for: pid)
            },
            set: { newValue in
                let pid = self.settings.selectedProviderID
                var saved = self.settings.savedProviders
                if let idx = saved.firstIndex(where: { $0.id == pid }) {
                    saved[idx].baseURL = newValue
                } else {
                    saved.append(SettingsStore.SavedProvider(id: pid, name: pid.capitalized, baseURL: newValue, apiKey: "", models: []))
                }
                self.settings.savedProviders = saved
            }
        )
    }

    private var apiKeyBinding: Binding<String> {
        Binding(
            get: {
                let pid = self.settings.selectedProviderID
                return self.settings.getAPIKey(for: pid) ?? ""
            },
            set: { newValue in
                let pid = self.settings.selectedProviderID
                var keys = self.settings.providerAPIKeys
                keys[pid] = newValue
                self.settings.providerAPIKeys = keys
            }
        )
    }

    private var modelBinding: Binding<String> {
        Binding(
            get: {
                let pid = self.settings.selectedProviderID
                let m = self.settings.selectedModelByProvider[pid] ?? ""
                return m.isEmpty ? self.defaultModel(for: pid) : m
            },
            set: { newValue in
                let pid = self.settings.selectedProviderID
                var models = self.settings.selectedModelByProvider
                models[pid] = newValue
                self.settings.selectedModelByProvider = models
            }
        )
    }

    private var activeSystemPromptBinding: Binding<String> {
        Binding(
            get: {
                let custom = self.settings.dictationCustomPromptText
                return custom.isEmpty ? "你是一个专业的语音文本整理助手。请准确保留用户的全部原意，自动去除语音中的'嗯'、'啊'、'那个'、'就是'等口语语气词，修正错别字并输出规范流畅的中文。" : custom
            },
            set: { newValue in
                self.settings.dictationCustomPromptText = newValue
            }
        )
    }

    private func defaultBaseURL(for providerID: String) -> String {
        switch providerID {
        case "ollama": return "http://192.168.7.136:11434"
        case "deepseek": return "https://api.deepseek.com"
        case "openrouter": return "https://openrouter.ai/api/v1"
        case "groq": return "https://api.groq.com/openai/v1"
        case "openai": return "https://api.openai.com/v1"
        default: return "http://localhost:11434"
        }
    }

    private func defaultModel(for providerID: String) -> String {
        switch providerID {
        case "ollama": return "qwen2.5:7b"
        case "deepseek": return "deepseek-chat"
        case "openrouter": return "openai/gpt-4o-mini"
        case "groq": return "llama-3.3-70b-versatile"
        case "openai": return "gpt-4o-mini"
        default: return "default"
        }
    }

    private func testLLMConnection() {
        self.isTestingConnection = true
        self.connectionMessage = nil
        self.connectionSuccess = nil

        let providerID = self.settings.selectedProviderID
        let baseURL = self.baseURLBinding.wrappedValue
        let apiKey = self.apiKeyBinding.wrappedValue
        let model = self.modelBinding.wrappedValue

        Task {
            let config = LLMClient.Config(
                messages: [
                    ["role": "system", "content": "You are a test assistant. Answer with OK."],
                    ["role": "user", "content": "Hello"]
                ],
                model: model,
                baseURL: baseURL,
                apiKey: apiKey,
                streaming: false
            )
            do {
                _ = try await LLMClient.shared.call(config)
                await MainActor.run {
                    self.isTestingConnection = false
                    self.connectionSuccess = true
                    self.connectionMessage = "Connection verified".loc
                }
            } catch {
                await MainActor.run {
                    self.isTestingConnection = false
                    self.connectionSuccess = false
                    self.connectionMessage = "\("Connection failed".loc): \(error.localizedDescription)"
                }
            }
        }
    }

    private func runSandboxTest() {
        self.isTestingSandbox = true
        self.sandboxOutput = ""

        let prompt = self.activeSystemPromptBinding.wrappedValue
        let text = self.sandboxInput
        let model = self.modelBinding.wrappedValue
        let baseURL = self.baseURLBinding.wrappedValue
        let apiKey = self.apiKeyBinding.wrappedValue

        Task {
            let config = LLMClient.Config(
                messages: [
                    ["role": "system", "content": prompt],
                    ["role": "user", "content": text]
                ],
                model: model,
                baseURL: baseURL,
                apiKey: apiKey,
                streaming: false
            )
            do {
                let response = try await LLMClient.shared.call(config)
                await MainActor.run {
                    self.isTestingSandbox = false
                    self.sandboxOutput = response.content
                }
            } catch {
                await MainActor.run {
                    self.isTestingSandbox = false
                    self.sandboxOutput = "Error: \(error.localizedDescription)"
                }
            }
        }
    }
}
