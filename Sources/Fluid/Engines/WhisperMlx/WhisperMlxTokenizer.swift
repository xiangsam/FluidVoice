import Foundation

// MARK: - Whisper tokenizers
//
// Two formats are supported:
// 1. OpenAI "multilingual.tiktoken" (large-v3-turbo repos): decode-only is
//    enough — we look up special tokens ("<|startoftranscript|>" etc.) by
//    matching the base64 encoding of the literal, and decode pieces by rank.
// 2. HuggingFace tokenizer.json BPE (whisper-small-asr repos): handled by
//    MlxSttTokenizerJson (GPT-2 byte-level BPE).

final class WhisperMlxTokenizer {

    private enum Impl {
        case tiktoken([Int: Data]) // rank -> raw bytes
        case bpe(MlxSttTokenizerJson)
    }

    private let impl: Impl

    init(tiktokenFile: URL) throws {
        let text = try String(contentsOf: tiktokenFile, encoding: .utf8)
        var ranks: [Int: Data] = [:]
        var firstLine = true
        // Line format: "<base64-token> <rank>". First line may be a JSON header
        // ("{\"vocab\": {\"<b64>\": <rank>, ...}}").
        for line in text.split(separator: "\n") {
            if firstLine, line.hasPrefix("{") {
                firstLine = false
                continue
            }
            firstLine = false
            let parts = line.split(separator: " ")
            guard parts.count >= 2 else { continue }
            let b64 = String(parts[0])
            let rank = Int(parts[1])
            guard let rank, let data = Data(base64Encoded: b64) else { continue }
            ranks[rank] = data
        }
        self.impl = .tiktoken(ranks)
    }

    init(tokenizerJson: URL) throws {
        self.impl = .bpe(try MlxSttTokenizerJson(contentsOf: tokenizerJson))
    }

    // MARK: - Special token lookup

    /// Rank of a special token literal (e.g. "<|startoftranscript|>").
    func specialID(_ literal: String) -> Int? {
        switch self.impl {
        case .tiktoken(let ranks):
            let b64 = Data(literal.utf8).base64EncodedString()
            for (rank, bytes) in ranks where bytes == Data(literal.utf8) {
                return rank
            }
            _ = b64
            return nil
        case .bpe(let tok):
            return nil // bpe path decodes specials via the BPE model directly
        }
    }

    /// All special-token ids (whisper control tokens range, 50257..50363).
    static var suppressSet: Set<Int> {
        var set = Set(50257...50363)
        set.insert(WhisperTokens.blank)
        return set
    }

    /// Decode generated token ids to text (skipping control tokens).
    func decode(_ ids: [Int], skipSpecialTokens: Bool = true) -> String {
        switch self.impl {
        case .tiktoken(let ranks):
            var bytes = Data()
            for id in ids {
                if skipSpecialTokens, id >= 50257 { continue }
                if let data = ranks[id] {
                    bytes.append(data)
                }
            }
            return String(data: bytes, encoding: .utf8) ?? ""
        case .bpe(let tok):
            return tok.decode(ids, skipSpecialTokens: skipSpecialTokens)
        }
    }
}
