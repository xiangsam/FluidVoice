//
//  String+Localization.swift
//  FluidVoice
//

import Foundation

public extension String {
    /// Returns the localized string based on the current app language setting
    var loc: String {
        LocalizationManager.shared.localized(self)
    }

    /// Alias for `.loc`
    var localized: String {
        self.loc
    }

    /// Localized string with format arguments
    func locFormat(_ arguments: CVarArg...) -> String {
        let format = LocalizationManager.shared.localized(self)
        return String(format: format, arguments: arguments)
    }
}
