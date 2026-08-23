import Foundation
import MLX

// MARK: - Whisper log-mel frontend (mlx_audio.stt.models.whisper.audio)
//
// hanning(400) / hop 160, centered STFT (reflect pad), power spectrum,
// slaney-normalized mel filterbank on the LIBROSA (slaney) mel scale,
// log10(max(1e-10)) -> clamp(max-8) -> (x + 4) / 4.
// Output layout: [1, nFrames, nMels] (channels-last, ready for the conv stem).

final class WhisperMlxMel {

    static let sampleRate = 16_000
    static let hop = 160
    static let chunkSamples = 480_000 // 30s
    static let nFrames = 3_000

    private let spec: WhisperMlxSpec
    private let nFFT = 400
    private let nFreq = 201

    private let window: MLXArray
    private let filterbank: MLXArray // [nMels, 201]

    init(spec: WhisperMlxSpec) {
        self.spec = spec
        // Symmetric Hann: 0.5 (1 - cos(2 pi n / (N-1)))
        let denom = Float(self.nFFT - 1)
        self.window = MLXArray((0..<self.nFFT).map { i -> Float in
            0.5 * (1.0 - cosf(2.0 * .pi * Float(i) / denom))
        })
        self.filterbank = Self.createFilterbank(nMels: spec.nMels, nFFT: self.nFFT, nFreq: self.nFreq)
    }

    /// audio [16k samples] -> [1, 3000, nMels] (30s window, padded/trimmed).
    func compute(audio: [Float]) -> MLXArray? {
        guard !audio.isEmpty else { return nil }
        let x = MLXArray(audio)
        // pad/trim to 480000
        let padded: MLXArray
        if x.size >= Self.chunkSamples {
            padded = x[0..<Self.chunkSamples]
        } else {
            let pad = zeros([Self.chunkSamples - x.size])
            padded = concatenated([x, pad])
        }

        // centered STFT with reflect padding
        let padLen = self.nFFT / 2
        let prefixRev = Array(padded[1..<(padLen + 1)].asArray(Float.self).reversed())
        let suffixRev = Array(padded[(padded.size - 1 - padLen)..<(padded.size - 1)].asArray(Float.self).reversed())
        let st = concatenated([MLXArray(prefixRev), padded, MLXArray(suffixRev)])

        let numFrames = 1 + (st.size - self.nFFT) / Self.hop
        var frames = [MLXArray]()
        frames.reserveCapacity(numFrames)
        for i in 0..<numFrames {
            let start = i * Self.hop
            frames.append(st[start..<(start + self.nFFT)] * self.window)
        }
        let strided = stacked(frames, axis: 0) // [numFrames, 400]
        let specArr = rfft(strided, n: self.nFFT, axis: 1) // [numFrames, 201]
        let mag = abs(specArr)
        let power = mag * mag
        let magnitudes = power[0..<(numFrames - 1), 0...] // drop last frame

        let melSpec = matmul(magnitudes, self.filterbank.T) // [F, nMels]
        var logSpec = log10(maximum(melSpec, 1e-10))
        let dynamicMax = max(logSpec).item(Float.self) - 8.0
        logSpec = maximum(logSpec, MLXArray(dynamicMax))
        logSpec = (logSpec + 4.0) / 4.0
        // Crop to the 3000-frame window and flip to [T, C]
        let t = min(logSpec.shape[0], Self.nFrames)
        let cropped = logSpec[0..<t, 0...]
        if t < Self.nFrames {
            let pad = zeros([Self.nFrames - t, self.spec.nMels])
            return concatenated([cropped, pad], axis: 0).expandedDimensions(axis: 0)
        }
        return cropped.expandedDimensions(axis: 0)
    }

    /// librosa mel filterbank on the slaney scale with slaney norm.
    static func createFilterbank(nMels: Int, nFFT: Int, nFreq: Int) -> MLXArray {
        let fMax = 8_000.0

        // slaney hz<->mel (mel_scale=None path in mlx_audio.utils.mel_filters)
        func hzToMel(_ hz: Double) -> Double {
            let fSp = 200.0 / 3.0
            let mels = hz / fSp
            let minLogHz = 1_000.0
            let minLogMel = minLogHz / fSp
            let logstep = log(6.4) / 27.0
            return hz >= minLogHz ? minLogMel + log(hz / minLogHz) / logstep : mels
        }
        func melToHz(_ mel: Double) -> Double {
            let fSp = 200.0 / 3.0
            let minLogHz = 1_000.0
            let minLogMel = minLogHz / fSp
            let logstep = log(6.4) / 27.0
            if mel >= minLogMel {
                return minLogHz * exp(logstep * (mel - minLogMel))
            }
            return fSp * mel
        }

        let mMin = hzToMel(0)
        let mMax = hzToMel(fMax)
        let melPts = (0..<(nMels + 2)).map { i -> Double in
            melToHz(mMin + Double(i) * (mMax - mMin) / Double(nMels + 1))
        }

        var fftFreqs = [Double](repeating: 0, count: nFreq)
        for i in 0..<nFreq {
            fftFreqs[i] = Double(i) * 16_000.0 / Double(nFFT)
        }

        var flat = [Float](repeating: 0, count: nMels * nFreq)
        for i in 0..<nMels {
            let fLeft = melPts[i], fCenter = melPts[i + 1], fRight = melPts[i + 2]
            for j in 0..<nFreq {
                let f = fftFreqs[j]
                let up = (f - fLeft) / (fCenter - fLeft)
                let down = (fRight - f) / (fRight - fCenter)
                var v = max(0, min(up, down))
                let width = fRight - fLeft
                if width > 0 { v *= 2.0 / width }
                flat[i * nFreq + j] = Float(v)
            }
        }
        return MLXArray(flat, [nMels, nFreq])
    }
}
