//
//  AISettingsView.swift
//  fluid
//
//  Extracted from ContentView.swift to reduce monolithic architecture.
//  Created: 2025-12-14
//

import SwiftUI

// MARK: - Connection Status Enum

enum AIConnectionStatus {
    case unknown, testing, success, failed
}

enum PromptEditorMode: Identifiable, Equatable {
    case defaultPrompt(mode: SettingsStore.PromptMode)
    case newPrompt(prefillMode: SettingsStore.PromptMode)
    case edit(promptID: String)
    case privateAI

    var id: String {
        switch self {
        case let .defaultPrompt(mode): return "default:\(mode.rawValue)"
        case let .newPrompt(prefillMode): return "new:\(prefillMode.rawValue)"
        case let .edit(promptID): return "edit:\(promptID)"
        case .privateAI: return "privateAI"
        }
    }

    var isDefault: Bool {
        if case .defaultPrompt = self { return true }
        return false
    }

    var isPrivateAI: Bool {
        if case .privateAI = self { return true }
        return false
    }

    var editingPromptID: String? {
        if case let .edit(promptID) = self { return promptID }
        return nil
    }

    var isNewPrompt: Bool {
        if case .newPrompt = self { return true }
        return false
    }

    var mode: SettingsStore.PromptMode? {
        switch self {
        case let .defaultPrompt(mode): return mode
        case let .newPrompt(prefillMode): return prefillMode
        case .edit: return nil
        case .privateAI: return .dictate
        }
    }
}

enum ModelSortOption: String, CaseIterable, Identifiable {
    case provider = "Provider"
    case accuracy = "Accuracy"
    case speed = "Speed"

    var id: String { self.rawValue }

    var localizedTitle: String {
        switch self {
        case .provider: return "Provider".loc
        case .accuracy: return "Accuracy".loc
        case .speed: return "Speed".loc
        }
    }
}

enum SpeechProviderFilter: String, CaseIterable, Identifiable {
    case all = "All"
    case cloud = "Cloud"
    case apple = "Apple"
    case openai = "OpenAI"
    case qwen = "Qwen"

    var id: String { self.rawValue }

    var localizedTitle: String {
        switch self {
        case .all: return "All".loc
        case .cloud: return "Cloud".loc
        case .apple: return "Apple"
        case .openai: return "OpenAI"
        case .qwen: return "Qwen"
        }
    }
}

