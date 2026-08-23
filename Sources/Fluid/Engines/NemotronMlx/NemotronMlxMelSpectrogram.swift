import Foundation
import MLX

// Mel frontend for Nemotron 3.5 ASR: mirrors mlx_audio nemotron_asr/audio.py
// log_mel_spectrogram:
// - pre-emphasis 0.97, reflect pad n_fft/2
// - symmetric Hann(400) center-padded to n_fft (56 zeros each side)
// - power spectrum |rfft|^2 (mag_power = 2.0)
// - librosa slaney mel filterbank (128 x 257), log(x + 2^-24)
// - normalize: "NA" -> no normalization
// Returns [1, T, 128] float32.
final class NemotronMlxMelSpectrogram {

    private let spec: NemotronMlxSpec

    private let winLength: Int
    private let hop: Int
    private let nFFT: Int
    private let nFreq: Int
    private let nMel: Int

    private let window: MLXArray // length nFFT (symmetric hann, center padded)
    private let filterbank: MLXArray // [nMel, nFreq] float32 (librosa slaney)
    private let preemph: Float
    private let logGuard: Float

    init(spec: NemotronMlxSpec) {
        self.spec = spec
        self.nFFT = spec.nFFT
        self.winLength = spec.winLength
        self.hop = spec.hopLength
        self.nFreq = spec.nFFT / 2 + 1
        self.nMel = spec.features
        self.preemph = spec.preemph
        self.logGuard = spec.logGuard

        // Symmetric Hann (denominator = N-1), then center-pad to n_fft.
        var raw = [Float](repeating: 0, count: self.winLength)
        let denom = Float(self.winLength - 1)
        for i in 0..<self.winLength {
            raw[i] = 0.5 * (1.0 - cosf(2.0 * .pi * Float(i) / denom))
        }
        let leftPad = (self.nFFT - self.winLength) / 2
        let rightPad = self.nFFT - self.winLength - leftPad
        self.window = MLXArray(
            [Float](repeating: 0, count: leftPad) + raw + [Float](repeating: 0, count: rightPad))

        self.filterbank = Self.createFilterbank(
            sr: spec.sampleRate, nFFT: self.nFFT, nMel: self.nMel, nFreq: self.nFreq)
    }

    /// [1, T, nMel] float32.
    func computeMLX(audio: [Float]) -> MLXArray? {
        let x = MLXArray(audio)
        if x.size == 0 { return nil }

        // pre-emphasis
        var y: MLXArray
        if self.preemph > 0 {
            let head = x[0..<1]
            let tail = x[1...] - self.preemph * x[0..<(x.size - 1)]
            y = concatenated([head, tail])
        } else {
            y = x
        }

        // reflect pad (center=True)
        let pad = self.nFFT / 2
        let prefixRev = Array(y[1..<(pad + 1)].asArray(Float.self).reversed())
        let suffixRev = Array(y[(y.size - 1 - pad)..<(y.size - 1)].asArray(Float.self).reversed())
        let padded = concatenated([MLXArray(prefixRev), y, MLXArray(suffixRev)])

        // num_frames = 1 + (N - n_fft) // hop  (padded length N+2*pad)
        let t = 1 + (padded.size - self.nFFT) / self.hop
        guard t > 0 else { return nil }

        var frames = [MLXArray]()
        frames.reserveCapacity(t)
        for i in 0..<t {
            let start = i * self.hop
            let frame = padded[start..<(start + self.nFFT)] * self.window
            frames.append(frame)
        }
        let framed = stacked(frames, axis: 0) // [T, 512]
        let specArr = rfft(framed, n: self.nFFT, axis: 1) // [T, 257] complex64

        // power spectrum |X|^2
        let view = specArr.view(dtype: .float32) // [T, 514]
        let re = view[0..., .stride(from: 0, to: 2 * self.nFreq, by: 2)]
        let im = view[0..., .stride(from: 1, to: 2 * self.nFreq, by: 2)]
        let power = re * re + im * im // [T, nFreq]

        // mel: [nMel, nFreq] @ [nFreq, T]
        let melSpec = matmul(self.filterbank, power.T) // [nMel, T]
        let logSpec = log(melSpec + self.logGuard) // [nMel, T]

        return logSpec.T.expandedDimensions(axis: 0) // [1, T, nMel]
    }

    // MARK: - librosa slaney mel filterbank (norm="slaney", mel_scale="slaney")

    private static func createFilterbank(sr: Int, nFFT: Int, nMel: Int, nFreq: Int) -> MLXArray {
        let fMax = Double(sr) / 2.0

        // librosa hz_to_mel / mel_to_hz with htk=false (Slaney piecewise)
        func hzToMel(_ hz: Double) -> Double {
            let fSp = 200.0 / 3.0
            if hz < 1000.0 { return hz / fSp }
            let minLogMel = 1000.0 / fSp
            let logStep = log(6.4) / 27.0
            return minLogMel + log(hz / 1000.0) / logStep
        }
        func melToHz(_ mel: Double) -> Double {
            let fSp = 200.0 / 3.0
            if mel < 15.0 { return mel * fSp }
            let minLogMel = 1000.0 / fSp
            let logStep = log(6.4) / 27.0
            return 1000.0 * exp(logStep * (mel - minLogMel))
        }

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
