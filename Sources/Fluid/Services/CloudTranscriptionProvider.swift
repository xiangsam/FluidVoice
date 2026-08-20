//
//  CloudTranscriptionProvider.swift
//  FluidVoice
//

import Foundation

enum CloudSTTType: String, CaseIterable, Identifiable, Codable {
    case openRouter = "openrouter"
    case openAI = "openai"
    case groq = "groq"
    case custom = "custom"

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .openRouter: return "OpenRouter"
        case .openAI: return "OpenAI"
        case .groq: return "Groq"
        case .custom: return "Custom (OpenAI-Compatible)"
        }
    }

    var defaultBaseURL: String {
        switch self {
        case .openRouter: return "https://openrouter.ai/api/v1"
        case .openAI: return "https://api.openai.com/v1"
        case .groq: return "https://api.groq.com/openai/v1"
        case .custom: return ""
        }
    }

    var defaultModel: String {
        switch self {
        case .openRouter: return "openai/whisper-large-v3"
        case .openAI: return "whisper-1"
        case .groq: return "whisper-large-v3"
        case .custom: return "whisper-1"
        }
    }

    var recommendedModels: [String] {
        switch self {
        case .openRouter:
            return [
                "openai/whisper-large-v3",
                "openai/whisper-large-v3-turbo",
                "openai/whisper-small",
                "openai/whisper-base",
                "openai/whisper-tiny",
                "google/gemini-2.0-flash-exp:free",
                "google/gemini-flash-1.5-8b",
                "google/gemini-flash-1.5",
                "meta-llama/llama-3.2-11b-vision-instruct",
            ]
        case .openAI:
            return [
                "whisper-1",
            ]
        case .groq:
            return [
                "whisper-large-v3",
                "whisper-large-v3-turbo",
                "distil-whisper-large-v3-en",
            ]
        case .custom:
            return [
                "whisper-1",
                "whisper-large-v3",
                "whisper-large-v3-turbo",
            ]
        }
    }
}

enum CloudTranscriptionError: LocalizedError {
    case missingAPIKey
    case invalidEndpointURL
    case networkError(String)
    case serverError(statusCode: Int, message: String)
    case invalidResponseData
    case emptyTranscriptionResult

    var errorDescription: String? {
        switch self {
        case .missingAPIKey:
            return "Cloud STT API Key is not configured. Please enter your API Key in Voice Engine settings.".loc
        case .invalidEndpointURL:
            return "Invalid Cloud STT API Base URL.".loc
        case let .networkError(msg):
            return "\("Network error during cloud transcription:".loc) \(msg)"
        case let .serverError(code, msg):
            return "\("Cloud STT API error (HTTP".loc) \(code)): \(msg)"
        case .invalidResponseData:
            return "Invalid response data received from cloud STT provider.".loc
        case .emptyTranscriptionResult:
            return "No transcription returned from cloud STT provider.".loc
        }
    }
}

final class CloudTranscriptionProvider: TranscriptionProvider {
    let type: CloudSTTType
    private let settings: SettingsStore

    init(type: CloudSTTType, settings: SettingsStore = SettingsStore.shared) {
        self.type = type
        self.settings = settings
    }

    var name: String {
        switch self.type {
        case .openRouter: return "OpenRouter Cloud STT"
        case .openAI: return "OpenAI Cloud STT"
        case .groq: return "Groq Cloud STT"
        case .custom: return "Custom Cloud STT"
        }
    }

    var isAvailable: Bool {
        true
    }

    var isReady: Bool {
        let key = self.resolvedAPIKey()
        return !key.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    var prefersNativeFileTranscription: Bool {
        true
    }

    var shouldClearCacheAfterCancellation: Bool {
        false
    }

    func modelsExistOnDisk() -> Bool {
        true
    }

    func clearCache() async throws {}

    func prepare(progressHandler: ((ModelPreparationProgress) -> Void)?) async throws {
        guard self.isReady else {
            throw CloudTranscriptionError.missingAPIKey
        }
        progressHandler?(.loading)
    }

    // MARK: - API Key & URL Resolution

    func resolvedAPIKey() -> String {
        switch self.type {
        case .openRouter:
            let key = self.settings.cloudSTTOpenRouterAPIKey
            if !key.isEmpty { return key }
            // Fallback to AI Enhancement OpenRouter API Key if available
            return self.settings.getAPIKey(for: "openrouter") ?? ""
        case .openAI:
            let key = self.settings.cloudSTTOpenAIAPIKey
            if !key.isEmpty { return key }
            return self.settings.getAPIKey(for: "openai") ?? ""
        case .groq:
            let key = self.settings.cloudSTTGroqAPIKey
            if !key.isEmpty { return key }
            return self.settings.getAPIKey(for: "groq") ?? ""
        case .custom:
            return self.settings.cloudSTTCustomAPIKey
        }
    }

    func resolvedBaseURL() -> String {
        let rawURL: String
        switch self.type {
        case .openRouter:
            rawURL = self.settings.cloudSTTOpenRouterBaseURL.isEmpty ? self.type.defaultBaseURL : self.settings.cloudSTTOpenRouterBaseURL
        case .openAI:
            rawURL = self.settings.cloudSTTOpenAIBaseURL.isEmpty ? self.type.defaultBaseURL : self.settings.cloudSTTOpenAIBaseURL
        case .groq:
            rawURL = self.settings.cloudSTTGroqBaseURL.isEmpty ? self.type.defaultBaseURL : self.settings.cloudSTTGroqBaseURL
        case .custom:
            rawURL = self.settings.cloudSTTCustomBaseURL
        }
        return rawURL.trimmingCharacters(in: .whitespacesAndNewlines).trimmingCharacters(in: CharacterSet(charactersIn: "/"))
    }

    func resolvedModel() -> String {
        let model: String
        switch self.type {
        case .openRouter:
            model = self.settings.cloudSTTOpenRouterModel
        case .openAI:
            model = self.settings.cloudSTTOpenAIModel
        case .groq:
            model = self.settings.cloudSTTGroqModel
        case .custom:
            model = self.settings.cloudSTTCustomModel
        }
        let trimmed = model.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? self.type.defaultModel : trimmed
    }

    private var endpointURL: URL? {
        let base = self.resolvedBaseURL()
        guard !base.isEmpty else { return nil }

        // Construct standard /audio/transcriptions endpoint
        let endpointString: String
        if base.hasSuffix("/audio/transcriptions") {
            endpointString = base
        } else {
            endpointString = "\(base)/audio/transcriptions"
        }
        return URL(string: endpointString)
    }

    // MARK: - Transcription Methods

    func transcribe(_ samples: [Float]) async throws -> ASRTranscriptionResult {
        guard !samples.isEmpty else {
            return ASRTranscriptionResult(text: "")
        }

        let wavData = try self.encodeTo16BitWAV(samples: samples, sampleRate: 16000)
        let text = try await self.sendAudioData(wavData, fileName: "audio.wav", mimeType: "audio/wav")

        return ASRTranscriptionResult(
            text: text,
            confidence: 1.0
        )
    }

    func transcribeFinal(_ samples: [Float]) async throws -> ASRTranscriptionResult {
        try await self.transcribe(samples)
    }

    func transcribeStreaming(_ samples: [Float]) async throws -> ASRTranscriptionResult {
        // Cloud APIs are typically batch request based. Return empty during live stream preview to save quota
        ASRTranscriptionResult(text: "", confidence: 1.0)
    }

    func transcribeDictionaryTraining(_ samples: [Float]) async throws -> ASRTranscriptionResult {
        try await self.transcribe(samples)
    }

    func transcribeFile(at fileURL: URL) async throws -> ASRTranscriptionResult {
        let audioData = try Data(contentsOf: fileURL)
        let fileName = fileURL.lastPathComponent
        let mimeType = self.mimeType(for: fileURL.pathExtension)

        let text = try await self.sendAudioData(audioData, fileName: fileName, mimeType: mimeType)
        return ASRTranscriptionResult(
            text: text,
            confidence: 1.0
        )
    }

    // MARK: - HTTP Multipart Request

    private func sendAudioData(_ audioData: Data, fileName: String, mimeType: String) async throws -> String {
        guard let url = self.endpointURL else {
            throw CloudTranscriptionError.invalidEndpointURL
        }

        let apiKey = self.resolvedAPIKey().trimmingCharacters(in: .whitespacesAndNewlines)
        guard !apiKey.isEmpty else {
            throw CloudTranscriptionError.missingAPIKey
        }

        let model = self.resolvedModel()
        let language = self.settings.cloudSTTLanguage

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")

        if self.type == .openRouter {
            request.setValue("FluidVoice macOS", forHTTPHeaderField: "HTTP-Referer")
            request.setValue("FluidVoice macOS App", forHTTPHeaderField: "X-Title")
        }

        let boundary = "Boundary-\(UUID().uuidString)"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")

        var body = Data()

        // 1. model parameter
        body.appendMultipartField(name: "model", value: model, boundary: boundary)

        // 2. language parameter (if specified and not "auto")
        if language != "auto" && !language.isEmpty {
            body.appendMultipartField(name: "language", value: language, boundary: boundary)
        }

        // 3. response_format = json
        body.appendMultipartField(name: "response_format", value: "json", boundary: boundary)

        // 4. file parameter
        body.appendMultipartFile(name: "file", fileName: fileName, mimeType: mimeType, fileData: audioData, boundary: boundary)

        // End boundary
        body.append("--\(boundary)--\r\n".data(using: .utf8)!)

        request.httpBody = body
        request.timeoutInterval = 60.0

        let (data, response): (Data, URLResponse)
        do {
            (data, response) = try await URLSession.shared.data(for: request)
        } catch {
            throw CloudTranscriptionError.networkError(error.localizedDescription)
        }

        guard let httpResponse = response as? HTTPURLResponse else {
            throw CloudTranscriptionError.invalidResponseData
        }

        guard (200...299).contains(httpResponse.statusCode) else {
            let errorBody = String(data: data, encoding: .utf8) ?? "Unknown server response"
            throw CloudTranscriptionError.serverError(statusCode: httpResponse.statusCode, message: errorBody)
        }

        // Parse JSON response: {"text": "..."}
        guard let jsonObject = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let text = jsonObject["text"] as? String else {
            // Try fallback plain text response
            if let plain = String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines), !plain.isEmpty {
                return plain
            }
            throw CloudTranscriptionError.invalidResponseData
        }

        return text.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // MARK: - Audio Conversion to 16-bit PCM WAV

    private func encodeTo16BitWAV(samples: [Float], sampleRate: Int) throws -> Data {
        var data = Data()

        let numChannels: Int16 = 1
        let bitsPerSample: Int16 = 16
        let byteRate: Int32 = Int32(sampleRate * Int(numChannels) * Int(bitsPerSample / 8))
        let blockAlign: Int16 = numChannels * (bitsPerSample / 8)
        let subchunk2Size: Int32 = Int32(samples.count * 2)
        let chunkSize: Int32 = 36 + subchunk2Size

        // RIFF header
        data.append("RIFF".data(using: .utf8)!)
        data.append(withUnsafeBytes(of: chunkSize.littleEndian) { Data($0) })
        data.append("WAVE".data(using: .utf8)!)

        // fmt chunk
        data.append("fmt ".data(using: .utf8)!)
        let subchunk1Size: Int32 = 16
        data.append(withUnsafeBytes(of: subchunk1Size.littleEndian) { Data($0) })
        let audioFormat: Int16 = 1 // PCM
        data.append(withUnsafeBytes(of: audioFormat.littleEndian) { Data($0) })
        data.append(withUnsafeBytes(of: numChannels.littleEndian) { Data($0) })
        let sampleRateInt32: Int32 = Int32(sampleRate)
        data.append(withUnsafeBytes(of: sampleRateInt32.littleEndian) { Data($0) })
        data.append(withUnsafeBytes(of: byteRate.littleEndian) { Data($0) })
        data.append(withUnsafeBytes(of: blockAlign.littleEndian) { Data($0) })
        data.append(withUnsafeBytes(of: bitsPerSample.littleEndian) { Data($0) })

        // data chunk
        data.append("data".data(using: .utf8)!)
        data.append(withUnsafeBytes(of: subchunk2Size.littleEndian) { Data($0) })

        // PCM Samples (Float -> Int16)
        for sample in samples {
            let clamped = max(-1.0, min(1.0, sample))
            let int16Sample = Int16(clamped * 32767.0)
            data.append(withUnsafeBytes(of: int16Sample.littleEndian) { Data($0) })
        }

        return data
    }

    private func mimeType(for ext: String) -> String {
        switch ext.lowercased() {
        case "wav": return "audio/wav"
        case "mp3": return "audio/mpeg"
        case "m4a": return "audio/m4a"
        case "ogg": return "audio/ogg"
        case "flac": return "audio/flac"
        case "webm": return "audio/webm"
        default: return "application/octet-stream"
        }
    }
}

// MARK: - Multipart Data Helpers

private extension Data {
    mutating func appendMultipartField(name: String, value: String, boundary: String) {
        var fieldString = "--\(boundary)\r\n"
        fieldString += "Content-Disposition: form-data; name=\"\(name)\"\r\n\r\n"
        fieldString += "\(value)\r\n"
        if let data = fieldString.data(using: .utf8) {
            self.append(data)
        }
    }

    mutating func appendMultipartFile(name: String, fileName: String, mimeType: String, fileData: Data, boundary: String) {
        var headerString = "--\(boundary)\r\n"
        headerString += "Content-Disposition: form-data; name=\"\(name)\"; filename=\"\(fileName)\"\r\n"
        headerString += "Content-Type: \(mimeType)\r\n\r\n"
        if let headerData = headerString.data(using: .utf8) {
            self.append(headerData)
        }
        self.append(fileData)
        if let trailing = "\r\n".data(using: .utf8) {
            self.append(trailing)
        }
    }
}
