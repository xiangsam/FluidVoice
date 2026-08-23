import Foundation

/// Parallel chunked downloader for MLX model weights.
///
/// HuggingFace serves weights over CDNs supporting HTTP Range requests
/// (`accept-ranges: bytes`). A single-stream `URLSession.data` download is
/// slow, especially via hf-mirror → huggingface.co redirects. This downloader
/// splits large files into fixed-size chunks, fetches them concurrently, and
/// assembles the file — the same technique `hf_transfer` uses under the hood
/// for `hf` CLI, but in pure Swift with zero external dependencies.
enum MlxModelDownloader {

    /// Chunk size per Range request (8 MiB — CDN friendly).
    static let chunkSize = 8 * 1024 * 1024
    /// Maximum number of concurrent chunk requests.
    static let concurrency = 8
    /// Files at or below this size go through a single request.
    static let singleRequestThreshold = 8 * 1024 * 1024

    /// Downloads `url` to `destination`, reporting `progress` (0…1).
    ///
    /// Writes to `destination.part` first and atomically renames it into place
    /// only after every byte arrived, so an interrupted or cancelled download
    /// never leaves a half-written model at the real path.
    static func downloadFile(
        from url: URL,
        to destination: URL,
        progress: (@Sendable (Double) -> Void)? = nil
    ) async throws {
        let fm = FileManager.default
        try fm.createDirectory(
            at: destination.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )

        let partURL = destination.appendingPathExtension("part")
        // Clean any stale partial from a previous interrupted attempt.
        if fm.fileExists(atPath: partURL.path) {
            try? fm.removeItem(at: partURL)
        }

        defer {
            // On any failure path below, discard the partial file so the model
            // directory never contains a broken weight file.
            if fm.fileExists(atPath: partURL.path) {
                try? fm.removeItem(at: partURL)
            }
        }

        // Learn the remote size (HEAD first; Range probe as a fallback).
        let size = try await remoteFileSize(url: url)
        guard size > 0 else {
            throw URLError(.zeroByteResource)
        }

        // Small files: single request is simpler and equally fast.
        if size <= Int64(Self.singleRequestThreshold) {
            let data = try await fetchRange(url: url, start: 0, end: UInt64(size) - 1, retries: 4)
            try data.write(to: partURL)
            try Self.atomicReplace(part: partURL, at: destination, fm: fm)
            progress?(1.0)
            return
        }

        // Large files: parallel range chunks, bounded by a semaphore so the
        // CDN never sees more than `concurrency` in-flight requests.
        let total = UInt64(size)
        let chunkCount = Int((total + UInt64(Self.chunkSize) - 1) / UInt64(Self.chunkSize))

        if !fm.fileExists(atPath: partURL.path) {
            fm.createFile(atPath: partURL.path, contents: nil)
        }
        let handle = try FileHandle(forWritingTo: partURL)
        defer { try? handle.close() }

        let progressLock = NSLock()
        let semaphore = AsyncSemaphore(value: Self.concurrency)

        try await withThrowingTaskGroup(of: Void.self) { group in
            for chunk in 0..<chunkCount {
                let start = UInt64(chunk) * UInt64(Self.chunkSize)
                let end = min(start + UInt64(Self.chunkSize), total) - 1
                group.addTask {
                    try await semaphore.withPermit {
                        let data = try await Self.fetchRange(
                            url: url, start: start, end: end, retries: 4
                        )
                        progressLock.lock()
                        do {
                            try handle.seek(toOffset: start)
                            try handle.write(contentsOf: data)
                        } catch {
                            progressLock.unlock()
                            throw error
                        }
                        progressLock.unlock()
                    }
                }
            }
        }

        try handle.close()
        try Self.atomicReplace(part: partURL, at: destination, fm: fm)
        progress?(1.0)
    }

    /// Moves the fully-downloaded `.part` file over the destination atomically.
    private static func atomicReplace(part: URL, at destination: URL, fm: FileManager) throws {
        if fm.fileExists(atPath: destination.path) {
            // Remove the old (possibly stale) file so rename never fails.
            try? fm.removeItem(at: destination)
        }
        try fm.moveItem(at: part, to: destination)
    }

    // MARK: - Range fetch

    /// Fetches `bytes=start-end` from `url`, retrying transient failures.
    private static func fetchRange(
        url: URL, start: UInt64, end: UInt64, retries: Int
    ) async throws -> Data {
        var lastError: Error = URLError(.unknown)
        for attempt in 0..<max(retries, 1) {
            do {
                var request = URLRequest(url: url)
                request.timeoutInterval = 180
                request.setValue("bytes=\(start)-\(end)", forHTTPHeaderField: "Range")
                request.setValue(
                    "Mozilla/5.0 (Macintosh; Apple Mac OS X) FluidVoice",
                    forHTTPHeaderField: "User-Agent"
                )
                let (data, response) = try await URLSession.shared.data(for: request)
                guard let http = response as? HTTPURLResponse else {
                    throw URLError(.cannotParseResponse)
                }
                if http.statusCode == 200, data.count > Int(end - start + 1) {
                    // CDN ignored Range and sent the whole body; slice the tail.
                    return data.suffix(Int(end - start + 1))
                }
                if http.statusCode == 206 || http.statusCode == 200 {
                    if data.isEmpty { throw URLError(.zeroByteResource) }
                    return data
                }
                throw URLError(.badServerResponse)
            } catch {
                lastError = error
                if attempt < retries - 1 {
                    try await Task.sleep(nanoseconds: 1_500_000_000)
                }
            }
        }
        throw lastError
    }

    /// Validates that `model.safetensors` at `url` is complete: reads the
    /// 8-byte header length, parses the JSON header and checks the declared
    /// tensor byte range fits the actual file size. A half-written file from an
    /// interrupted download fails this check, so callers can treat it as
    /// missing and re-download.
    static func isValidSafetensors(at url: URL) -> Bool {
        guard let handle = try? FileHandle(forReadingFrom: url) else { return false }
        defer { try? handle.close() }
        guard let size = try? handle.seekToEnd(), size > 8 else { return false }
        // seekToEnd() moved the cursor to EOF — rewind before reading the header.
        do {
            try handle.seek(toOffset: 0)
        } catch {
            return false
        }
        guard let headerLenBytes = try? handle.read(upToCount: 8), headerLenBytes.count == 8 else {
            return false
        }
        var headerLength: UInt64 = 0
        for b in headerLenBytes.reversed() {
            headerLength = (headerLength << 8) | UInt64(b)
        }
        guard headerLength > 0, headerLength + 8 <= UInt64(size) else { return false }
        guard let headerData = try? handle.read(upToCount: Int(headerLength)) else { return false }
        guard let obj = try? JSONSerialization.jsonObject(with: headerData) as? [String: Any] else {
            return false
        }
        // The safetensors header records each tensor's byte offsets; verify the
        // last tensor's end falls inside the file so a truncated download is
        // detected even when the header itself parses fine.
        var tensorEnd: UInt64 = 0
        for (key, value) in obj {
            guard key != "__metadata__",
                  let entry = value as? [String: Any],
                  let begin = (entry["data_offsets"] as? [Int])?.first,
                  let offset = entry["data_offsets"] as? [Int], offset.count >= 2
            else { continue }
            tensorEnd = max(tensorEnd, UInt64(offset[1]))
            _ = begin
        }
        let headerEnd = 8 + headerLength
        guard tensorEnd > 0, headerEnd + tensorEnd <= UInt64(size) else { return false }
        return true
    }

    /// HEAD request to learn the remote file size; Range probe as fallback.
    private static func remoteFileSize(url: URL) async throws -> Int64 {
        var request = URLRequest(url: url)
        request.httpMethod = "HEAD"
        request.timeoutInterval = 30
        request.setValue(
            "Mozilla/5.0 (Macintosh; Apple Mac OS X) FluidVoice",
            forHTTPHeaderField: "User-Agent"
        )
        if let (_, response) = try? await URLSession.shared.data(for: request),
           let http = response as? HTTPURLResponse
        {
            if let len = http.value(forHTTPHeaderField: "Content-Length"),
               let value = Int64(len), value > 0
            {
                return value
            }
        }
        // Fall back to a Range probe parsing Content-Range total.
        var probe = URLRequest(url: url)
        probe.setValue("bytes=0-0", forHTTPHeaderField: "Range")
        probe.timeoutInterval = 30
        probe.setValue(
            "Mozilla/5.0 (Macintosh; Apple Mac OS X) FluidVoice",
            forHTTPHeaderField: "User-Agent"
        )
        let (_, response) = try await URLSession.shared.data(for: probe)
        guard let http = response as? HTTPURLResponse,
              let cr = http.value(forHTTPHeaderField: "Content-Range"),
              let slash = cr.split(separator: "/").last,
              let total = Int64(slash), total > 0
        else {
            throw URLError(.cannotParseResponse)
        }
        return total
    }
}

/// Async semaphore (actor-based) to bound concurrent chunk fetches.
private actor AsyncSemaphore {
    private var value: Int
    private var waiters: [CheckedContinuation<Void, Never>] = []

    init(value: Int) {
        self.value = value
    }

    func withPermit<T: Sendable>(_ body: @escaping () async throws -> T) async throws -> T {
        await acquire()
        defer { Task { await release() } }
        return try await body()
    }

    private func acquire() async {
        if self.value > 0 {
            self.value -= 1
            return
        }
        await withCheckedContinuation { continuation in
            self.waiters.append(continuation)
        }
    }

    private func release() {
        if let next = self.waiters.first {
            self.waiters.removeFirst()
            next.resume()
        } else {
            self.value += 1
        }
    }
}
