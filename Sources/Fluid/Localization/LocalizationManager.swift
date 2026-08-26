//
//  LocalizationManager.swift
//  MlxVoice
//

import Combine
import Foundation
import SwiftUI

public extension Notification.Name {
    static let appLanguageDidChange = Notification.Name("MlxVoiceAppLanguageDidChange")
}

public final class LocalizationManager: ObservableObject {
    public static let shared = LocalizationManager()

    private let userDefaultsKey = "app_preferred_language"
    private let lock = NSLock()
    private var _cachedLanguage: AppLanguage = .system

    @MainActor @Published public var currentLanguage: AppLanguage {
        didSet {
            self.lock.lock()
            self._cachedLanguage = self.currentLanguage
            self.lock.unlock()
            UserDefaults.standard.set(self.currentLanguage.rawValue, forKey: self.userDefaultsKey)
            NotificationCenter.default.post(name: .appLanguageDidChange, object: self.currentLanguage)
        }
    }

    private init() {
        let initialLang: AppLanguage
        if let saved = UserDefaults.standard.string(forKey: self.userDefaultsKey),
           let lang = AppLanguage(rawValue: saved)
        {
            initialLang = lang
        } else {
            initialLang = .system
        }
        self._cachedLanguage = initialLang
        self.currentLanguage = initialLang
    }

    /// Thread-safe active language check
    public var isChinese: Bool {
        self.lock.lock()
        let lang = self._cachedLanguage
        self.lock.unlock()

        switch lang {
        case .zhHans:
            return true
        case .en:
            return false
        case .system:
            if let pref = Locale.preferredLanguages.first?.lowercased() {
                return pref.hasPrefix("zh")
            }
            return false
        }
    }

    /// Thread-safe string localization lookup
    public func localized(_ key: String) -> String {
        guard self.isChinese else {
            return key
        }
        return ChineseDictionary.translations[key] ?? key
    }

    /// Thread-safe formatted localization
    public func localizedFormat(_ key: String, _ arguments: CVarArg...) -> String {
        let format = self.localized(key)
        return String(format: format, arguments: arguments)
    }
}
