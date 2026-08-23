import Foundation

struct PronunciationTextReplacement {
    let wordRange: ClosedRange<Int>
    let label: String
}

enum PronunciationTextReplacer {
    static func dictionaryLabels(
        from entries: [SettingsStore.CustomDictionaryEntry]
    ) -> [UUID: String] {
        Dictionary(
            entries.map { ($0.id, $0.replacement) },
            uniquingKeysWith: { _, last in last }
        )
    }

    static func applyingPronunciationReplacements(
        to text: String,
        wordTexts: [String],
        replacements: [PronunciationTextReplacement]
    ) -> String {
        let source = text as NSString
        var searchLocation = 0
        var ranges: [NSRange] = []
        let trimSet = CharacterSet.whitespacesAndNewlines.union(.punctuationCharacters)

        for wordText in wordTexts {
            let searchableWord = wordText.trimmingCharacters(in: trimSet)
            guard !searchableWord.isEmpty, searchLocation <= source.length else { return text }
            let searchRange = NSRange(location: searchLocation, length: source.length - searchLocation)
            let range = source.range(of: searchableWord, options: .caseInsensitive, range: searchRange)
            guard range.location != NSNotFound else { return text }
            ranges.append(range)
            searchLocation = NSMaxRange(range)
        }

        let edits = replacements.compactMap { replacement -> (NSRange, String)? in
            guard ranges.indices.contains(replacement.wordRange.lowerBound),
                  ranges.indices.contains(replacement.wordRange.upperBound)
            else { return nil }
            let start = ranges[replacement.wordRange.lowerBound].location
            let end = NSMaxRange(ranges[replacement.wordRange.upperBound])
            return (NSRange(location: start, length: end - start), replacement.label)
        }.sorted { $0.0.location > $1.0.location }

        let output = NSMutableString(string: text)
        for (range, label) in edits {
            output.replaceCharacters(in: range, with: label)
        }
        return output as String
    }
}
