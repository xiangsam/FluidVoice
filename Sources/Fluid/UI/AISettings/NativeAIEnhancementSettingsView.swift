//
//  NativeAIEnhancementSettingsView.swift
//  FluidVoice
//
//  Modern, card-styled native SwiftUI AI Enhancement View adhering to Apple HIG.
//

import SwiftUI

enum RefinementPreset: String, CaseIterable, Identifiable {
    case cleanFillers = "cleanFillers"
    case writtenText = "writtenText"
    case typography = "typography"
    case coding = "coding"
    case custom = "custom"

    var id: String { self.rawValue }

    var title: String {
        switch self {
        case .cleanFillers: return "去除语气词与杂音".loc
        case .writtenText: return "口语转标准书面语".loc
        case .typography: return "中英文排版与标点".loc
        case .coding: return "编程与代码专有词".loc
        case .custom: return "自定义提示词 (Prompt)".loc
        }
    }

    var subtitle: String {
        switch self {
        case .cleanFillers:
            return "自动过滤口语中的“嗯、啊、那个、就是”等词，准确保留全部原意，输出干净纯粹的文本。".loc
        case .writtenText:
            return "理顺语句逻辑，修复语病和逻辑停顿，适当补充标准标点，整理为连贯通顺的书面表达。".loc
        case .typography:
            return "在中英文与数字间自动补充标准空格，修正误标点与漏标点，排版整洁美观。".loc
        case .coding:
            return "精准保留技术术语、库名与变量名（camelCase、snake_case 等），输出开发笔记格式。".loc
        case .custom:
            return "自由编写定制化的 System Prompt，打造专属于您的 AI 语音处理助手。".loc
        }
    }

    var icon: String {
        switch self {
        case .cleanFillers: return "sparkles"
        case .writtenText: return "text.badge.checkmark"
        case .typography: return "character.bubble.fill"
        case .coding: return "curlybraces"
        case .custom: return "slider.horizontal.3"
        }
    }

    var iconColor: Color {
        switch self {
        case .cleanFillers: return Color.fluidGreen
        case .writtenText: return Color.blue
        case .typography: return Color.orange
        case .coding: return Color.purple
        case .custom: return Color.secondary
        }
    }

    var defaultPrompt: String {
        switch self {
        case .cleanFillers:
            return "你是一个专业的语音文本整理助手。请准确保留用户的全部原意，自动去除语音中的'嗯'、'啊'、'那个'、'就是'、'然后'等口语语气词和多余重复词，修正错别字，输出干净规范的中文。"
        case .writtenText:
            return "请将用户的口语表述整理为优雅、连贯的标准书面语，修复病句与逻辑停顿，适当补充规范标点符号，保持说话者的核心原意不变。"
        case .typography:
            return "请规范用户的文本排版：在中文字符与英文单词、数字之间自动添加标准空格，纠正漏标点和误标点，输出排版规范的中文。"
        case .coding:
            return "用户正在口述编程与技术内容。请精准保留所有的技术术语、库名、函数名与变量名（如 camelCase、snake_case），输出清晰简洁的开发笔记格式。"
        case .custom:
            return ""
        }
    }
}

struct NativeAIEnhancementSettingsView: View {
    @ObservedObject var viewModel: AIEnhancementSettingsViewModel
    @ObservedObject var settings: SettingsStore
    @ObservedObject var promptTest: DictationPromptTestCoordinator
    let theme: AppTheme

    @State private var selectedPreset: RefinementPreset = .cleanFillers
    @State private var isTestingConnection: Bool = false
    @State private var connectionMessage: String? = nil
    @State private var connectionSuccess: Bool? = nil

    @State private var playgroundInput: String = "呃，那个，我今天测试了一下Ollama模型，感觉速度还可以啊。"
    @State private var playgroundOutput: String = ""
    @State private var isTestingPlayground: Bool = false

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
                // 3. Preset Gallery
                self.presetGallerySection

                // 4. LLM Provider Settings
                self.providerCardSection

                // 5. Interactive Playground
                self.playgroundSection
            }
        }
        .padding(.horizontal, 4)
        .padding(.vertical, 8)
        .onAppear {
            self.detectCurrentPreset()
        }
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
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .fill(self.isAIEnabledBinding.wrappedValue ? self.theme.palette.accent.opacity(0.15) : Color.secondary.opacity(0.12))
                    .frame(width: 48, height: 48)
                Image(systemName: "brain.head.profile")
                    .font(.system(size: 22))
                    .foregroundStyle(self.isAIEnabledBinding.wrappedValue ? self.theme.palette.accent : .secondary)
            }

            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 6) {
                    Text("Enable AI Post-Processing".loc)
                        .font(.headline)
                        .fontWeight(.semibold)

                    if self.isAIEnabledBinding.wrappedValue {
                        Text("Active".loc)
                            .font(.system(size: 10, weight: .bold))
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(Capsule().fill(Color.fluidGreen.opacity(0.2)))
                            .foregroundStyle(Color.fluidGreen)
                    }
                }

                Text(self.isAIEnabledBinding.wrappedValue ? "录音结束后将由大模型自动润色并输出完美文本".loc : "语音识别文字将直接秒级打字上屏，不经过大模型处理".loc)
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
                        .stroke(Color.secondary.opacity(0.18), lineWidth: 1)
                )
        )
    }

    // MARK: - 3. Preset Gallery Section
    private var presetGallerySection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Refinement Preset & Rules".loc)
                .font(.headline)
                .foregroundStyle(.primary)

            VStack(spacing: 8) {
                ForEach(RefinementPreset.allCases) { preset in
                    self.presetCard(for: preset)
                }
            }

            // Expanded Custom Prompt Editor
            if self.selectedPreset == .custom {
                VStack(alignment: .leading, spacing: 6) {
                    Text("Custom System Prompt:".loc)
                        .font(.caption)
                        .fontWeight(.medium)
                        .foregroundStyle(.secondary)

                    TextEditor(text: self.customPromptBinding)
                        .font(.system(.subheadline, design: .monospaced))
                        .frame(minHeight: 80)
                        .padding(6)
                        .background(RoundedRectangle(cornerRadius: 8).fill(Color(nsColor: .textBackgroundColor)))
                        .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.secondary.opacity(0.2), lineWidth: 1))
                }
                .padding(.top, 4)
            }
        }
    }

    private func presetCard(for preset: RefinementPreset) -> some View {
        let isSelected = self.selectedPreset == preset

        return Button {
            withAnimation(.easeInOut(duration: 0.15)) {
                self.selectedPreset = preset
                if preset != .custom {
                    self.settings.dictationCustomPromptText = preset.defaultPrompt
                }
                self.settings.setDictationPromptSelection(.default)
            }
        } label: {
            HStack(spacing: 12) {
                ZStack {
                    Circle()
                        .fill(preset.iconColor.opacity(0.14))
                        .frame(width: 32, height: 32)
                    Image(systemName: preset.icon)
                        .font(.system(size: 14, weight: .semibold))
                        .foregroundStyle(preset.iconColor)
                }

                VStack(alignment: .leading, spacing: 2) {
                    Text(preset.title)
                        .font(.system(size: 13, weight: isSelected ? .semibold : .regular))
                        .foregroundStyle(.primary)

                    Text(preset.subtitle)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                }

                Spacer()

                if isSelected {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundStyle(self.theme.palette.accent)
                        .font(.system(size: 17))
                } else {
                    Circle()
                        .strokeBorder(Color.secondary.opacity(0.35), lineWidth: 1.5)
                        .frame(width: 17, height: 17)
                }
            }
            .padding(12)
            .background(
                RoundedRectangle(cornerRadius: 10)
                    .fill(isSelected ? self.theme.palette.accent.opacity(0.06) : Color(nsColor: .controlBackgroundColor).opacity(0.6))
                    .overlay(
                        RoundedRectangle(cornerRadius: 10)
                            .stroke(isSelected ? self.theme.palette.accent.opacity(0.45) : Color.secondary.opacity(0.14), lineWidth: 1)
                    )
            )
        }
        .buttonStyle(.plain)
    }

    // MARK: - 4. LLM Provider Card Section
    private var providerCardSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("LLM Provider Configuration".loc)
                .font(.headline)
                .foregroundStyle(.primary)

            VStack(spacing: 12) {
                // Provider Segmented
                Picker("", selection: self.$settings.selectedProviderID) {
                    Text("Ollama (局域网)".loc).tag("ollama")
                    Text("DeepSeek").tag("deepseek")
                    Text("OpenRouter").tag("openrouter")
                    Text("Groq").tag("groq")
                    Text("OpenAI").tag("openai")
                }
                .pickerStyle(.segmented)

                Divider().opacity(0.3)

                // Simplified Provider Inputs
                if self.settings.selectedProviderID == "ollama" {
                    HStack {
                        Text("Ollama 地址".loc)
                            .font(.subheadline)
                            .frame(width: 110, alignment: .leading)
                        TextField("http://192.168.7.136:11434", text: self.baseURLBinding)
                            .textFieldStyle(.roundedBorder)
                    }
                    HStack {
                        Text("模型名称".loc)
                            .font(.subheadline)
                            .frame(width: 110, alignment: .leading)
                        TextField("qwen2.5:7b", text: self.modelBinding)
                            .textFieldStyle(.roundedBorder)
                    }
                } else {
                    HStack {
                        Text("API Key".loc)
                            .font(.subheadline)
                            .frame(width: 110, alignment: .leading)
                        SecureField("sk-...", text: self.apiKeyBinding)
                            .textFieldStyle(.roundedBorder)
                    }
                    HStack {
                        Text("模型名称".loc)
                            .font(.subheadline)
                            .frame(width: 110, alignment: .leading)
                        TextField(self.defaultModel(for: self.settings.selectedProviderID), text: self.modelBinding)
                            .textFieldStyle(.roundedBorder)
                    }
                }

                // Context Length & Memory Estimation
                VStack(alignment: .leading, spacing: 6) {
                    let aiEstimate = self.settings.estimatePrivateAIBreakdown()
                    HStack {
                        Text("润色上下文长度 (Context Limit):".loc)
                            .font(.subheadline)
                            .foregroundStyle(.primary)
                        Spacer()
                        Picker("", selection: self.$settings.privateAIContextTokenLimit) {
                            Text("2,048 Tokens (~0.5G 缓存)").tag(2048)
                            Text("4,096 Tokens (~1.0G 缓存 · 推荐)".loc).tag(4096)
                            Text("8,192 Tokens (~2.0G 缓存)").tag(8192)
                            Text("16,384 Tokens (~4.0G 缓存)").tag(16384)
                        }
                        .pickerStyle(.menu)
                        .controlSize(.small)
                        .frame(maxWidth: 250)
                    }

                    if self.settings.selectedProviderID == "ollama" {
                        HStack(spacing: 6) {
                            Circle().fill(aiEstimate.pressure == .safe ? Color.fluidGreen : Color.orange).frame(width: 6, height: 6)
                            Text("本地大模型预计内存占用: ~\(String(format: "%.1f", aiEstimate.totalGB)) GB (模型权重 4.2G + 上下文缓存 \(String(format: "%.1f", aiEstimate.contextKVCacheGB))G + 运行开销)".loc)
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }
                    }
                }
                .padding(.top, 4)

                // Test Connection Row
                HStack {
                    if let message = self.connectionMessage {
                        Text(message)
                            .font(.caption)
                            .foregroundStyle(self.connectionSuccess == true ? Color.fluidGreen : .red)
                    }

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
                            Text(self.isTestingConnection ? "正在连接...".loc : "测试连接".loc)
                        }
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    .disabled(self.isTestingConnection)
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

    // MARK: - 5. Interactive Playground Section
    private var playgroundSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Interactive Playground".loc)
                .font(.headline)
                .foregroundStyle(.primary)

            VStack(alignment: .leading, spacing: 12) {
                Text("输入一段口述草稿，点击“体验润色”即时预览 AI 优化效果：".loc)
                    .font(.caption)
                    .foregroundStyle(.secondary)

                HStack(alignment: .top, spacing: 10) {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("原始输入".loc)
                            .font(.caption)
                            .fontWeight(.medium)
                            .foregroundStyle(.secondary)
                        TextField("输入测试口语文本...", text: self.$playgroundInput)
                            .textFieldStyle(.roundedBorder)
                    }

                    Button {
                        self.runPlaygroundTest()
                    } label: {
                        HStack(spacing: 4) {
                            if self.isTestingPlayground {
                                ProgressView().controlSize(.small)
                            } else {
                                Image(systemName: "sparkles")
                            }
                            Text(self.isTestingPlayground ? "润色中...".loc : "体验润色".loc)
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.regular)
                    .disabled(self.isTestingPlayground || self.playgroundInput.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                    .padding(.top, 18)
                }

                if !self.playgroundOutput.isEmpty {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("AI 润色结果：".loc)
                            .font(.caption)
                            .fontWeight(.medium)
                            .foregroundStyle(Color.fluidGreen)

                        Text(self.playgroundOutput)
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

    private var customPromptBinding: Binding<String> {
        Binding(
            get: { self.settings.dictationCustomPromptText },
            set: { newValue in self.settings.dictationCustomPromptText = newValue }
        )
    }

    private func detectCurrentPreset() {
        let current = self.settings.dictationCustomPromptText.trimmingCharacters(in: .whitespacesAndNewlines)
        if current.isEmpty || current == RefinementPreset.cleanFillers.defaultPrompt {
            self.selectedPreset = .cleanFillers
        } else if current == RefinementPreset.writtenText.defaultPrompt {
            self.selectedPreset = .writtenText
        } else if current == RefinementPreset.typography.defaultPrompt {
            self.selectedPreset = .typography
        } else if current == RefinementPreset.coding.defaultPrompt {
            self.selectedPreset = .coding
        } else {
            self.selectedPreset = .custom
        }
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
                    self.connectionMessage = "连接测试成功".loc
                }
            } catch {
                await MainActor.run {
                    self.isTestingConnection = false
                    self.connectionSuccess = false
                    self.connectionMessage = "\("连接失败".loc): \(error.localizedDescription)"
                }
            }
        }
    }

    private func runPlaygroundTest() {
        self.isTestingPlayground = true
        self.playgroundOutput = ""

        let prompt = self.selectedPreset == .custom ? self.customPromptBinding.wrappedValue : self.selectedPreset.defaultPrompt
        let text = self.playgroundInput
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
                    self.isTestingPlayground = false
                    self.playgroundOutput = response.content
                }
            } catch {
                await MainActor.run {
                    self.isTestingPlayground = false
                    self.playgroundOutput = "错误: \(error.localizedDescription)"
                }
            }
        }
    }
}
