import Foundation
import MLX

// Mel frontend for GLM-ASR (Whisper-style), mirroring glmasr._preprocess_audio:
// - symmetric Hann(N_FFT=400), reflect pad n_fft/2 (stft center)
// - power spectrum |rfft|^2, drop last frame
// - mel filterbank: HTK mel scale + slaney normalization (librosa-style)
// - log10(max(spec, 1e-10)), clamp to max-8, then (x + 4) / 4
// Returns [T, 128] float32 (whisper channel layout [batch, seq, mels]).
final class GlmMlxMelSpectrogram {

    private let spec: GlmMlxSpec

    private let nFFT = 400
    private let hop = 160
    private let nFreq = 201 // 400/2 + 1
    private let nMel: Int

    private let window: MLXArray // symmetric Hann(400)
    private let filterbank: MLXArray // [128, 201] (htk mel, slaney norm)

    init(spec: GlmMlxSpec) {
        self.spec = spec
        self.nMel = spec.numMelBins

        // Symmetric Hann: denom = N-1 (mlx_audio.utils.hanning(periodic=False))
        let denom = Float(self.nFFT - 1)
        let raw = (0..<self.nFFT).map { i -> Float in
            0.5 * (1.0 - cosf(2.0 * .pi * Float(i) / denom))
        }
        self.window = MLXArray(raw)

        self.filterbank = Self.createFilterbank(
            sr: spec.sampleRate, nFFT: self.nFFT, nMel: self.nMel, nFreq: self.nFreq)
    }

    /// [1, T, 128] float32.
    func computeMLX(audio: [Float]) -> MLXArray? {
        let x = MLXArray(audio)
        if x.size == 0 { return nil }

        // reflect pad (stft center=True)
        let pad = self.nFFT / 2
        let prefixRev = Array(x[1..<(pad + 1)].asArray(Float.self).reversed())
        let suffixRev = Array(x[(x.size - 1 - pad)..<(x.size - 1)].asArray(Float.self).reversed())
        let padded = concatenated([MLXArray(prefixRev), x, MLXArray(suffixRev)])

        let tAll = 1 + (padded.size - self.nFFT) / self.hop
        guard tAll > 1 else { return nil }
        let t = tAll - 1 // drop last STFT frame

        var frames = [MLXArray]()
        frames.reserveCapacity(tAll)
        for i in 0..<tAll {
            let start = i * self.hop
            let frame = padded[start..<(start + self.nFFT)] * self.window
            frames.append(frame)
        }
        let framed = stacked(frames, axis: 0) // [tAll, 400]
        let specArr = rfft(framed, n: self.nFFT, axis: 1) // [tAll, 201] complex
        let mag = abs(specArr) // complex abs
        let power = mag * mag // [tAll, 201]
        let magnitudes = power[0..<t, 0...] // drop last frame

        let melSpec = matmul(magnitudes, self.filterbank.T) // [t, 128]
        let logSpec = log10(maximum(melSpec, 1e-10)) // [t, 128]
        let dynamicMax = max(logSpec).item(Float.self) - 8.0
        let clamped = maximum(logSpec, MLXArray(dynamicMax))
        return ((clamped + 4.0) / 4.0).expandedDimensions(axis: 0) // [1, t, 128]
    }

    // MARK: - HTK mel scale + slaney normalization (librosa filters.mel)

    private static func createFilterbank(sr: Int, nFFT: Int, nMel: Int, nFreq: Int) -> MLXArray {
        let fMax = Double(sr) / 2.0

        func hzToMel(_ hz: Double) -> Double { 2595.0 * log10(1.0 + hz / 700.0) }
        func melToHz(_ mel: Double) -> Double { 700.0 * (pow(10.0, mel / 2595.0) - 1.0) }

        let melMin = hzToMel(0)
        let melMax = hzToMel(fMax)
        let melPts = (0..<(nMel + 2)).map { i -> Double in
            melToHz(melMin + Double(i) * (melMax - melMin) / Double(nMel + 1))
        }

        var fftFreqs = [Double](repeating: 0, count: nFreq)
        for i in 0..<nFreq {
            fftFreqs[i] = Double(i) * Double(sr) / Double(nFFT)
        }

        var flat = [Float](repeating: 0, count: nMel * nFreq)
        for i in 0..<nMel {
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
        return MLXArray(flat, [nMel, nFreq])
    }
}
