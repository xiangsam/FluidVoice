//
//  AppLanguage.swift
//  MlxVoice
//

import Foundation

public enum AppLanguage: String, CaseIterable, Identifiable, Codable {
    case system = "system"
    case zhHans = "zh-Hans"
    case en = "en"

    public var id: String { rawValue }

    public var displayName: String {
        switch self {
        case .system:
            return "跟随系统 (System)"
        case .zhHans:
            return "简体中文 (Chinese)"
        case .en:
            return "English"
        }
    }

    public var shortName: String {
        switch self {
        case .system:
            return "自动"
        case .zhHans:
            return "中文"
        case .en:
            return "EN"
        }
    }
}
