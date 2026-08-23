import Foundation

// MARK: - MLX STT Model Catalog (card-based)
//
// A card is one concrete downloadable model: family x size x quantization.
// Cards are independently selectable / downloadable / unloadable, matching
// the Whisper model cards. Only the common quantization tiers are listed
// (bf16 + 8bit by default; families whose size already varies a lot stay at
// two tiers). New mlx-community ASR families are added here as their
// engine backends land.

/// Engine family / architecture of an MLX STT model.
enum MlxSttFamilyKind: String, CaseIterable, Identifiable, Hashable {
    case qwen3Asr = "qwen3-asr"
    case parakeetTdt = "parakeet-tdt"
    case nemotronAsr = "nemotron-asr"
    case glmAsr = "glm-asr"
    case funAsr = "fun-asr"
    case whisperMlx = "whisper-mlx"

    var id: String { self.rawValue }

    var displayName: String {
        switch self {
        case .qwen3Asr: return "Qwen3-ASR (千问语音)"
        case .parakeetTdt: return "Parakeet TDT (NVIDIA)"
        case .nemotronAsr: return "Nemotron ASR (NVIDIA)"
        case .glmAsr: return "GLM-ASR (智谱)"
        case .funAsr: return "Fun-ASR (通义)"
        case .whisperMlx: return "Whisper (MLX)"
        }
    }

    var detail: String {
        switch self {
        case .qwen3Asr: return "30 语言多语识别 · 阿里通义千问 ASR"
        case .parakeetTdt: return "FastConformer · TDT 解码 · NVIDIA"
        case .nemotronAsr: return "FastConformer · 流式 · NVIDIA"
        case .glmAsr: return "Whisper 编码器 + LLaMA · 多语"
        case .funAsr: return "SANM 编码器 + Qwen3 · 多语（含中文）"
        case .whisperMlx: return "OpenAI Whisper · MLX 架构 · 99 语"
        }
    }
}

/// Quantization tier of a card.
enum MlxSttQuant: String, Hashable {
    case bf16 = "bf16"
    case int8 = "8bit"

    var label: String { self.rawValue }

    var note: String {
        switch self {
        case .bf16: return "最高精度"
        case .int8: return "推荐 · 低内存高速度"
        }
    }
}

/// Per-card recommendation tier shown in the model card list. The app's own
/// judgment for every local engine: Fun-ASR is the best overall pick (esp.
/// Chinese mixed speech), Qwen3-ASR is the runner-up, Whisper supports 99
/// languages incl. Chinese (good for both), and Parakeet TDT is European-only
/// (it does NOT support Chinese at all).
enum MlxSttRecommendation: Int, Hashable, Comparable {
    case mustHave = 0   // 第一推荐（fun）
    case recommended = 1 // 第二推荐（qwen）
    case good = 2        // 可用替代（glm / nemotron / whisper 英文+中文）
    case notRecommended = 3 // 不支持中文（parakeet 欧洲语系专用）

    static func < (lhs: MlxSttRecommendation, rhs: MlxSttRecommendation) -> Bool {
        lhs.rawValue < rhs.rawValue
    }

    var label: String {
        switch self {
        case .mustHave: return "强推"
        case .recommended: return "推荐"
        case .good: return "可选"
        case .notRecommended: return "不支持中文"
        }
    }
}

/// One concrete downloadable model: `family × size × quant`.
struct MlxSttCard: Identifiable, Hashable {
    /// Directory-level identifier of this card inside its family,
    /// e.g. "0.6B-8bit". Combined with the family it forms `pathID`.
    let id: String
    let family: MlxSttFamilyKind
    /// Size / spec label inside the family, e.g. "0.6B".
    let specID: String
    /// Quantization tier. `nil` for families with a single precision.
    let quant: MlxSttQuant?
    /// Human readable card title, e.g. "Qwen3-ASR 0.6B".
    let title: String
    /// The exact mlx-community repository that feeds this card.
    let repo: String
    let sizeBytes: Int64
    let note: String?

    /// Unique key used for settings and on-disk layout: "qwen3-asr/0.6B-8bit".
    var pathID: String { "\(self.family.rawValue)/\(self.id)" }

    var quantLabel: String? { self.quant?.label }

    var sizeDescription: String {
        let gb = Double(self.sizeBytes) / 1_073_741_824
        return String(format: "%.2f GB", gb)
    }

    var footnote: String {
        var parts: [String] = []
        if let quant { parts.append(quant.note) }
        if let note { parts.append(note) }
        let base = parts.joined(separator: " · ")
        return base.isEmpty ? "自动匹配 \(self.repo)" : "\(base) · \(self.repo)"
    }
}

enum MlxSttCatalog {

    // MARK: - Qwen3-ASR (backend: Qwen3MlxEngine, live)

    static let qwen3AsrCards: [MlxSttCard] = [
        MlxSttCard(
            id: "0.6B-8bit",
            family: .qwen3Asr,
            specID: "0.6B",
            quant: .int8,
            title: "Qwen3-ASR 0.6B",
            repo: "mlx-community/Qwen3-ASR-0.6B-8bit",
            sizeBytes: 1_010_771_234,
            note: "低内存高速度"
        ),
        MlxSttCard(
            id: "0.6B-bf16",
            family: .qwen3Asr,
            specID: "0.6B",
            quant: .bf16,
            title: "Qwen3-ASR 0.6B",
            repo: "mlx-community/Qwen3-ASR-0.6B-bf16",
            sizeBytes: 1_569_435_907,
            note: "小模型最高精度"
        ),
        MlxSttCard(
            id: "1.7B-8bit",
            family: .qwen3Asr,
            specID: "1.7B",
            quant: .int8,
            title: "Qwen3-ASR 1.7B",
            repo: "mlx-community/Qwen3-ASR-1.7B-8bit",
            sizeBytes: 2_467_856_503,
            note: "高质量大模型"
        ),
        MlxSttCard(
            id: "1.7B-bf16",
            family: .qwen3Asr,
            specID: "1.7B",
            quant: .bf16,
            title: "Qwen3-ASR 1.7B",
            repo: "mlx-community/Qwen3-ASR-1.7B-bf16",
            sizeBytes: 4_080_707_826,
            note: "顶级质量"
        ),
    ]

    // MARK: - Parakeet TDT (backend: ParakeetMlxEngine, live)

    static let parakeetTdtCards: [MlxSttCard] = [
        MlxSttCard(
            id: "0.6B-v3",
            family: .parakeetTdt,
            specID: "0.6B v3",
            quant: .bf16,
            title: "Parakeet TDT 0.6B v3",
            repo: "mlx-community/parakeet-tdt-0.6b-v3",
            sizeBytes: 2_508_288_736,
            note: "25 语言 · 仅欧洲语系 · 不支持中文"
        ),
    ]

    // MARK: - Nemotron 3.5 ASR (backend: NemotronMlxEngine, live)

    static let nemotronAsrCards: [MlxSttCard] = [
        MlxSttCard(
            id: "0.6B-streaming",
            family: .nemotronAsr,
            specID: "0.6B",
            quant: .bf16,
            title: "Nemotron 3.5 ASR 0.6B",
            repo: "mlx-community/nemotron-3.5-asr-streaming-0.6b",
            sizeBytes: 1_276_058_836,
            note: "多语 · 支持中文 · 流式 FastConformer-RNNT"
        ),
    ]

    // MARK: - GLM-ASR (backend: GlmMlxEngine, live)

    static let glmAsrCards: [MlxSttCard] = [
        MlxSttCard(
            id: "0.6B-nano-8bit",
            family: .glmAsr,
            specID: "Nano",
            quant: .int8,
            title: "GLM-ASR Nano",
            repo: "mlx-community/GLM-ASR-Nano-2512-8bit",
            sizeBytes: 2_409_627_301,
            note: "智谱 · 高精度多语（含中文）"
        ),
    ]

    // MARK: - Fun-ASR (backend: FunMlxEngine, live)

    static let funAsrCards: [MlxSttCard] = [
        MlxSttCard(
            id: "0.6B-mlt-nano-8bit",
            family: .funAsr,
            specID: "MLT Nano",
            quant: .int8,
            title: "Fun-ASR MLT Nano",
            repo: "mlx-community/Fun-ASR-MLT-Nano-2512-8bit",
            sizeBytes: 1_546_273_485,
            note: "31 语言 · 中英混说 · 通义 Fun-ASR"
        ),
    ]

    // MARK: - Whisper (MLX architecture, replaces transcribe.cpp)

    static let whisperMlxCards: [MlxSttCard] = [
        MlxSttCard(
            id: "whisper-tiny-8bit",
            family: .whisperMlx,
            specID: "Tiny",
            quant: .int8,
            title: "Whisper Tiny",
            repo: "mlx-community/whisper-tiny-8bit",
            sizeBytes: 40_245_083,
            note: "超轻量 · 最快 · 99 语（含中文）"
        ),
        MlxSttCard(
            id: "whisper-small-8bit",
            family: .whisperMlx,
            specID: "Small",
            quant: .int8,
            title: "Whisper Small",
            repo: "mlx-community/whisper-small-8bit",
            sizeBytes: 258_100_000,
            note: "轻量平衡 · 99 语（含中文）"
        ),
        MlxSttCard(
            id: "whisper-large-v3-turbo-8bit",
            family: .whisperMlx,
            specID: "Large V3 Turbo",
            quant: .int8,
            title: "Whisper Large V3 Turbo",
            repo: "mlx-community/whisper-large-v3-turbo-8bit",
            sizeBytes: 863_659_156,
            note: "OpenAI 旗舰 · 99 语（含中文）· 高精度"
        ),
        MlxSttCard(
            id: "whisper-large-v3-8bit",
            family: .whisperMlx,
            specID: "Large V3",
            quant: .int8,
            title: "Whisper Large V3",
            repo: "mlx-community/whisper-large-v3-8bit",
            sizeBytes: 1_560_000_000,
            note: "OpenAI V3 非 Turbo · 99 语（含中文）· 占显存最大"
        ),
    ]

    /// Recommended tier for a local MLX card. The ordering reflects the app's
    /// own experience: Fun-ASR first (best Chinese mixed-speech accuracy),
    /// Qwen3-ASR second, Whisper supports 99 languages incl. Chinese, and the
    /// rest are good alternatives except Parakeet (European languages only).
    static func recommendation(for card: MlxSttCard) -> MlxSttRecommendation {
        switch card.family {
        case .funAsr:
            return .mustHave
        case .qwen3Asr:
            return .recommended
        case .whisperMlx, .glmAsr, .nemotronAsr:
            return .good
        case .parakeetTdt:
            return .notRecommended
        }
    }

    /// Cards sorted by recommendation tier, then by title.
    static var cardsByRecommendation: [MlxSttCard] {
        Self.cards.sorted { lhs, rhs in
            let l = Self.recommendation(for: lhs)
            let r = Self.recommendation(for: rhs)
            if l != r { return l < r }
            return lhs.title < rhs.title
        }
    }

    /// Maps legacy whisper SpeechModels to their MLX whisper card pathIDs.
    static func whisperCardID(for model: SettingsStore.SpeechModel) -> String? {
        switch model {
        case .whisperLargeTurbo: return "whisper-mlx/whisper-large-v3-turbo-8bit"
        case .whisperSmall: return "whisper-mlx/whisper-small-asr-4bit"
        default: return nil
        }
    }

    // MARK: - All cards (ordered for UI)

    static let cards: [MlxSttCard] =
        Self.qwen3AsrCards
        + Self.parakeetTdtCards
        + Self.nemotronAsrCards
        + Self.glmAsrCards
        + Self.funAsrCards
        + Self.whisperMlxCards

    // MARK: - Lookup

    static func card(pathID: String) -> MlxSttCard? {
        self.cards.first { $0.pathID == pathID }
    }

    static func cards(for family: MlxSttFamilyKind) -> [MlxSttCard] {
        self.cards.filter { $0.family == family }
    }

    /// Default selected card (also used as legacy-default fallback).
    /// Fun-ASR is the app's strongest pick for Chinese mixed speech.
    static let defaultCardPathID = "fun-asr/0.6B-mlt-nano-8bit"
}
