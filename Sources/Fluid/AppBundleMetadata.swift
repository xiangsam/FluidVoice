import Foundation

enum AppSupportPaths {
    static let folderName = "MlxVoice"
    static let legacyFolderName = "FluidVoice"

    /// Rename `Application Support/FluidVoice` → `MlxVoice` once, so already-downloaded
    /// MLX models and dictionaries keep working after the rebrand.
    static func migrateLegacyDirectoryIfNeeded() {
        let fileManager = FileManager.default
        guard let base = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first else {
            return
        }
        let current = base.appendingPathComponent(folderName, isDirectory: true)
        let legacy = base.appendingPathComponent(legacyFolderName, isDirectory: true)
        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: legacy.path, isDirectory: &isDirectory), isDirectory.boolValue else {
            return
        }
        guard !fileManager.fileExists(atPath: current.path) else { return }
        try? fileManager.moveItem(at: legacy, to: current)
    }
}

extension Bundle {
    var fluidAppDisplayName: String {
        let displayName = self.object(forInfoDictionaryKey: "CFBundleDisplayName") as? String
        let bundleName = self.object(forInfoDictionaryKey: "CFBundleName") as? String
        return [displayName, bundleName]
            .compactMap { $0?.trimmingCharacters(in: .whitespacesAndNewlines) }
            .first { !$0.isEmpty } ?? "MlxVoice"
    }
}
