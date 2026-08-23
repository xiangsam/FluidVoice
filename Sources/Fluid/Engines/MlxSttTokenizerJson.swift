import Foundation

// MARK: - tokenizer.json BPE helper (HuggingFace format)
//
// Supports the LLaMA-style byte-level BPE tokenizers shipped by GLM-ASR /
// Fun-ASR repositories (tokenizer.json with model.type == "BPE").
// Special tokens are handled as atomic pieces.

final class MlxSttTokenizerJson {
    let vocabToID: [String: Int]
    let idToVocab: [Int: String]
    let merges: [String: Int]
    let specialTokens: [Int: String]
    private let byteEncoder: [UInt8: UInt32]
    private let byteDecoder: [UInt32: UInt8]

    init(contentsOf url: URL) throws {
        let data = try Data(contentsOf: url)
        let root = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        let model = root?["model"] as? [String: Any]
        guard let rawVocab = model?["vocab"] as? [String: Int] else {
            throw NSError(domain: "TokenizerJson", code: -1, userInfo: [NSLocalizedDescriptionKey: "Missing BPE vocab"])
        }
        self.vocabToID = rawVocab
        var idToV: [Int: String] = [:]
        for (k, v) in rawVocab { idToV[v] = k }
        self.idToVocab = idToV

        var mergeMap: [String: Int] = [:]
        if let rawMerges = (model?["merges"] as? [Any]) {
            for (i, m) in rawMerges.enumerated() {
                // Newer tokenizer.json stores merges as ["left","right"] pairs;
                // older ones as single "left right" strings.
                let key: String
                if let s = m as? String {
                    key = s
                } else if let pair = m as? [String], pair.count == 2 {
                    key = "\(pair[0]) \(pair[1])"
                } else {
                    continue
                }
                mergeMap[key] = i
            }
        }
        self.merges = mergeMap

        var special: [Int: String] = [:]
        if let added = root?["added_tokens"] as? [[String: Any]] {
            for t in added {
                if let id = t["id"] as? Int, let content = t["content"] as? String {
                    special[id] = content
                }
            }
        }
        self.specialTokens = special

        // gpt2 bytes_to_unicode table (byte-level BPE pre-tokenizer)
        var bs = [Int]()
        bs.append(contentsOf: 33...126)
        bs.append(contentsOf: 161...172)
        bs.append(contentsOf: 174...255)
        var cs = bs
        var n = 0
        for b in 0...255 where !bs.contains(b) {
            bs.append(b)
            cs.append(256 + n)
            n += 1
        }
        let enc = Dictionary(uniqueKeysWithValues: zip(bs.map { UInt8($0) }, cs.map { UInt32($0) }))
        self.byteEncoder = enc
        var dec: [UInt32: UInt8] = [:]
        for (b, c) in enc { dec[c] = b }
        self.byteDecoder = dec
    }

    /// Encode text to token ids (byte-level BPE) with special-token scanning.
    func encode(_ text: String) -> [Int] {
        let pieces: [String] =
            self.specialTokens.isEmpty
            ? [text]
            : self.splitSpecial(text)

        var out: [Int] = []
        for piece in pieces {
            if let sid = self.specialTokenID(for: piece) {
                out.append(sid)
                continue
            }
            out.append(contentsOf: self.encodeRegular(piece))
        }
        return out
    }

    func decode(_ ids: [Int], skipSpecialTokens: Bool = true) -> String {
        var pieces: [String] = []
        for id in ids {
            if let s = self.specialTokens[id] {
                if !skipSpecialTokens { pieces.append(s) }
                continue
            }
            if let p = self.idToVocab[id] {
                pieces.append(p)
            }
        }
        let raw = pieces.joined()
        var bytes = [UInt8]()
        for scalar in raw.unicodeScalars {
            if let b = self.byteDecoder[scalar.value] {
                bytes.append(b)
            } else if scalar.value < 256 {
                bytes.append(UInt8(scalar.value))
            }
        }
        return String(bytes: bytes, encoding: .utf8) ?? raw
    }

    // MARK: - internals

    private func specialTokenID(for piece: String) -> Int? {
        // tokenizer.json special tokens are full pieces (e.g. "<|assistant|>")
        if let id = self.vocabToID[piece] { return id }
        // added_tokens fallback
        for (id, content) in self.specialTokens where content == piece {
            return id
        }
        return nil
    }

    private func splitSpecial(_ text: String) -> [String] {
        let specials = Array(Set(self.specialTokens.values)).sorted { $0.count > $1.count }
        guard !specials.isEmpty else { return [text] }

        var result: [String] = []
        var current = ""
        var idx = text.startIndex
        while idx < text.endIndex {
            var matched: String?
            for sp in specials {
                if text[idx...].hasPrefix(sp) {
                    matched = sp
                    break
                }
            }
            if let sp = matched {
                if !current.isEmpty { result.append(current); current = "" }
                result.append(sp)
                idx = text.index(idx, offsetBy: sp.count)
            } else {
                current.append(text[idx])
                idx = text.index(after: idx)
            }
        }
        if !current.isEmpty { result.append(current) }
        return result
    }

    private func encodeRegular(_ text: String) -> [Int] {
        let chars = Array(text.utf8).map { byte -> String in
            let cp = self.byteEncoder[byte] ?? UInt32(byte)
            return String(Character(UnicodeScalar(cp)!))
        }
        guard !chars.isEmpty else { return [] }
        var segments = chars
        while segments.count > 1 {
            var bestRank = Int.max
            var bestIndex = -1
            for i in 0..<(segments.count - 1) {
                let key = "\(segments[i]) \(segments[i + 1])"
                if let rank = self.merges[key], rank < bestRank {
                    bestRank = rank
                    bestIndex = i
                }
            }
            guard bestIndex >= 0 else { break }
            segments[bestIndex] = segments[bestIndex] + segments[bestIndex + 1]
            segments.remove(at: bestIndex + 1)
        }
        return segments.compactMap { self.vocabToID[$0] }
    }
}
