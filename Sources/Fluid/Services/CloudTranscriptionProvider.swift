//
//  CloudTranscriptionProvider.swift
//  FluidVoice
//

import Foundation

public enum CloudSTTType: String, CaseIterable, Identifiable {
    case openRouter = "openrouter"
    case openAI = "openai"
    case groq = "groq"
    case custom = "custom"

    public var id: String { self.rawValue }

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
        case .openRouter: return "openai/gpt-4o-mini-transcribe"
        case .openAI: return "whisper-1"
        case .groq: return "whisper-large-v3"
        case .custom: return "whisper-1"
        }
    }

    var recommendedModels: [String] {
        self.modelPresets.map(\.id)
    }

    var modelPresets: [CloudSTTModelItem] {
        switch self {
        case .openRouter:
            return [
                CloudSTTModelItem(
                    id: "openai/gpt-4o-mini-transcribe",
                    name: "OpenAI: GPT-4o Mini Transcribe",
                    vendor: "OpenAI",
                    tag: "🔥 最热门",
                    priceHint: "$1.25/M",
                    isPopular: true
                ),
                CloudSTTModelItem(
                    id: "openai/gpt-4o-transcribe",
                    name: "OpenAI: GPT-4o Transcribe",
                    vendor: "OpenAI",
                    tag: "💎 旗舰高精",
                    priceHint: "$2.50/M",
                    isPopular: true
                ),
                CloudSTTModelItem(
                    id: "mistralai/voxtral-mini-transcribe",
                    name: "Mistral: Voxtral Mini Transcribe",
                    vendor: "Mistral",
                    tag: "⚡️ 极速推荐",
                    priceHint: "$0.003",
                    isPopular: true
                ),
                CloudSTTModelItem(
                    id: "nvidia/nemotron-3.5-asr-streaming-multilingual-0.6b",
                    name: "NVIDIA: Nemotron 3.5 ASR Streaming 0.6B",
                    vendor: "NVIDIA",
                    tag: "💰 超高性价比",
                    priceHint: "$0.000003"
                ),
                CloudSTTModelItem(
                    id: "mistralai/voxtral-small-24b-2507-stt",
                    name: "Mistral: Voxtral Small 24B 2507 STT",
                    vendor: "Mistral",
                    tag: "24B 大模型",
                    priceHint: "$0.00005"
                ),
                CloudSTTModelItem(
                    id: "mistralai/voxtral-mini-3b-2507",
                    name: "Mistral: Voxtral Mini 3B 2507",
                    vendor: "Mistral",
                    priceHint: "$0.000017"
                ),
                CloudSTTModelItem(
                    id: "qwen/qwen3-asr-1.7b",
                    name: "Qwen: Qwen3 ASR 1.7B",
                    vendor: "Alibaba Qwen",
                    tag: "🎯 中文极佳",
                    priceHint: "$0.000008"
                ),
                CloudSTTModelItem(
                    id: "qwen/qwen3-asr-0.6b",
                    name: "Qwen: Qwen3 ASR 0.6B",
                    vendor: "Alibaba Qwen",
                    tag: "中文极速",
                    priceHint: "$0.000003"
                ),
                CloudSTTModelItem(
                    id: "openai/gpt-transcribe",
                    name: "OpenAI: GPT Transcribe",
                    vendor: "OpenAI",
                    priceHint: "$0.0045"
                ),
                CloudSTTModelItem(
                    id: "fish-audio/transcribe-1",
                    name: "Fish Audio: Transcribe 1",
                    vendor: "Fish Audio",
                    priceHint: "$0.0001"
                ),
                CloudSTTModelItem(
                    id: "x-ai/grok-stt-1.0",
                    name: "SpaceXAI: Grok STT 1.0",
                    vendor: "xAI",
                    tag: "Grok 引擎",
                    priceHint: "$0.10"
                ),
                CloudSTTModelItem(
                    id: "deepgram/nova-3",
                    name: "Deepgram: Nova-3",
                    vendor: "Deepgram",
                    tag: "工业级快速",
                    priceHint: "from $0.0043"
                ),
                CloudSTTModelItem(
                    id: "microsoft/mai-transcribe-1.5",
                    name: "Microsoft: MAI-Transcribe 1.5",
                    vendor: "Microsoft",
                    priceHint: "$0.36"
                ),
                CloudSTTModelItem(
                    id: "nvidia/parakeet-tdt-0.6b-v3",
                    name: "NVIDIA: Parakeet TDT 0.6B v3",
                    vendor: "NVIDIA",
                    priceHint: "$0.0015"
                ),
                CloudSTTModelItem(
                    id: "qwen/qwen3-asr-flash-2026-02-10",
                    name: "Qwen: Qwen3 ASR Flash",
                    vendor: "Alibaba Qwen",
                    priceHint: "$0.000035"
                ),
                CloudSTTModelItem(
                    id: "google/chirp-3",
                    name: "Google: Chirp 3",
                    vendor: "Google",
                    tag: "Google 旗舰",
                    priceHint: "$0.016"
                ),
                CloudSTTModelItem(
                    id: "openai/whisper-large-v3-turbo",
                    name: "OpenAI: Whisper Large V3 Turbo",
                    vendor: "OpenAI",
                    tag: "经典 Turbo",
                    priceHint: "$0.000003"
                ),
                CloudSTTModelItem(
                    id: "openai/whisper-large-v3",
                    name: "OpenAI: Whisper Large V3",
                    vendor: "OpenAI",
                    tag: "经典高精度",
                    priceHint: "$0.000008"
                ),
                CloudSTTModelItem(
                    id: "openai/whisper-1",
                    name: "OpenAI: Whisper 1",
                    vendor: "OpenAI",
                    priceHint: "$0.006"
                ),
            ]
        case .openAI:
            return [
                CloudSTTModelItem(id: "whisper-1", name: "OpenAI Whisper 1", vendor: "OpenAI", tag: "标准转录", priceHint: "$0.006/min"),
                CloudSTTModelItem(id: "gpt-4o-mini-transcribe", name: "GPT-4o Mini Transcribe", vendor: "OpenAI", tag: "极速轻量"),
                CloudSTTModelItem(id: "gpt-4o-transcribe", name: "GPT-4o Transcribe", vendor: "OpenAI", tag: "旗舰精度"),
            ]
        case .groq:
            return [
                CloudSTTModelItem(id: "whisper-large-v3", name: "Whisper Large V3", vendor: "Groq", tag: "极速 LPU 加速"),
                CloudSTTModelItem(id: "whisper-large-v3-turbo", name: "Whisper Large V3 Turbo", vendor: "Groq", tag: "超低延迟"),
                CloudSTTModelItem(id: "distil-whisper-large-v3-en", name: "Distil-Whisper Large V3 (English)", vendor: "Groq", tag: "英文专精"),
            ]
        case .custom:
            return [
                CloudSTTModelItem(id: "whisper-1", name: "Whisper 1", vendor: "Custom"),
                CloudSTTModelItem(id: "whisper-large-v3", name: "Whisper Large V3", vendor: "Custom"),
                CloudSTTModelItem(id: "whisper-large-v3-turbo", name: "Whisper Large V3 Turbo", vendor: "Custom"),
            ]
        }
    }
}

public struct CloudSTTModelItem: Identifiable, Hashable {
    public let id: String
    public let name: String
    public let vendor: String
    public let tag: String?
    public let priceHint: String?
    public let isPopular: Bool

    public init(
        id: String,
        name: String,
        vendor: String,
        tag: String? = nil,
        priceHint: String? = nil,
        isPopular: Bool = false
    ) {
        self.id = id
        self.name = name
        self.vendor = vendor
        self.tag = tag
        self.priceHint = priceHint
        self.isPopular = isPopular
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
        case let .serverError(statusCode, msg):
            return "\("Cloud STT API error (HTTP".loc) \(statusCode)): \(msg)"
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

    // MARK: - HTTP Multipart Request with Auto-Fallback

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
            request.setValue("https://github.com/xiangsam/FluidVoice", forHTTPHeaderField: "HTTP-Referer")
            request.setValue("FluidVoice macOS App", forHTTPHeaderField: "X-Title")
        }

        let boundary = "Boundary-\(UUID().uuidString)"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")

        var body = Data()
        body.appendMultipartField(name: "model", value: model, boundary: boundary)

        if language != "auto" && !language.isEmpty {
            body.appendMultipartField(name: "language", value: language, boundary: boundary)
        }

        body.appendMultipartField(name: "response_format", value: "json", boundary: boundary)
        body.appendMultipartFile(name: "file", fileName: fileName, mimeType: mimeType, fileData: audioData, boundary: boundary)
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

        guard let jsonObject = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let text = jsonObject["text"] as? String else {
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

        data.append("RIFF".data(using: .utf8)!)
        data.append(withUnsafeBytes(of: chunkSize.littleEndian) { Data($0) })
        data.append("WAVE".data(using: .utf8)!)

        data.append("fmt ".data(using: .utf8)!)
        let subchunk1Size: Int32 = 16
        let audioFormat: Int16 = 1
        data.append(withUnsafeBytes(of: subchunk1Size.littleEndian) { Data($0) })
        data.append(withUnsafeBytes(of: audioFormat.littleEndian) { Data($0) })
        data.append(withUnsafeBytes(of: numChannels.littleEndian) { Data($0) })
        data.append(withUnsafeBytes(of: Int32(sampleRate).littleEndian) { Data($0) })
        data.append(withUnsafeBytes(of: byteRate.littleEndian) { Data($0) })
        data.append(withUnsafeBytes(of: blockAlign.littleEndian) { Data($0) })
        data.append(withUnsafeBytes(of: bitsPerSample.littleEndian) { Data($0) })

        data.append("data".data(using: .utf8)!)
        data.append(withUnsafeBytes(of: subchunk2Size.littleEndian) { Data($0) })

        for sample in samples {
            let clamped = max(-1.0, min(1.0, sample))
            let intSample = Int16(clamped * 32767.0)
            data.append(withUnsafeBytes(of: intSample.littleEndian) { Data($0) })
        }

        return data
    }

    private func mimeType(for pathExtension: String) -> String {
        switch pathExtension.lowercased() {
        case "wav": return "audio/wav"
        case "m4a": return "audio/m4a"
        case "mp3": return "audio/mp3"
        case "ogg": return "audio/ogg"
        case "flac": return "audio/flac"
        case "aac": return "audio/aac"
        default: return "audio/wav"
        }
    }
}

// MARK: - Data Multipart Extension

private extension Data {
    mutating func appendMultipartField(name: String, value: String, boundary: String) {
        let fieldString = "--\(boundary)\r\nContent-Disposition: form-data; name=\"\(name)\"\r\n\r\n\(value)\r\n"
        if let data = fieldString.data(using: .utf8) {
            self.append(data)
        }
    }

    mutating func appendMultipartFile(name: String, fileName: String, mimeType: String, fileData: Data, boundary: String) {
        let header = "--\(boundary)\r\nContent-Disposition: form-data; name=\"\(name)\"; filename=\"\(fileName)\"\r\nContent-Type: \(mimeType)\r\n\r\n"
        if let headerData = header.data(using: .utf8) {
            self.append(headerData)
        }
        self.append(fileData)
        self.append("\r\n".data(using: .utf8)!)
    }
}
