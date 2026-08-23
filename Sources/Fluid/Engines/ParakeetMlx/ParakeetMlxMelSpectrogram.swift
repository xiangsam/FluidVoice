import Foundation
import MLX

// Mel frontend for Parakeet TDT, mirroring parakeet_mlx/audio.py::
// get_logmel(PreprocessArgs) exactly, computed with the MLX ops that the
// Python reference uses (identical numerics):
// - pre-emphasis 0.97, reflect pad n_fft/2
// - periodic Hann(win_length), zero-padded window to n_fft (512)
// - rfft, complex view |re| + |im| (257 bins incl. Nyquist), ^mag_power 2
// - librosa slaney mel filterbank (128 x 257), log(x + 1e-5)
// - per_feature normalize across frames: (x - mean) / (std + 1e-5)
// Returns [1, T, features] float32.
final class ParakeetMlxMelSpectrogram {

    private let spec: ParakeetMlxSpec

    private let winLength: Int
    private let hop: Int
    private let nFFT: Int
    private let nFreq: Int // 257 = nFFT/2 + 1
    private let nMel: Int

    private let window: MLXArray // length nFFT (periodic hann + zero pad)
    private let filterbank: MLXArray // [nMel, nFreq] float32 (librosa slaney)
    private let preemph: Float
    private let magPower: Float

    init(spec: ParakeetMlxSpec) {
        self.spec = spec
        self.nFFT = spec.nFFT
        self.winLength = spec.winLength
        self.hop = spec.hopLength
        self.nFreq = spec.nFFT / 2 + 1
        self.nMel = spec.features
        self.preemph = spec.preemph
        self.magPower = spec.magPower

        // Periodic Hann window (np.hanning(n + 1)[:-1]) zero-padded to nFFT.
        var w = [Float](repeating: 0, count: self.nFFT)
        let n = Float(self.winLength)
        for i in 0..<self.winLength {
            w[i] = 0.5 * (1.0 - cosf(2.0 * .pi * Float(i) / n))
        }
        self.window = MLXArray(w)

        // librosa.filters.mel(sr=16000, n_fft=512, n_mels=128, fmin=0, fmax=8000, norm="slaney")
        self.filterbank = Self.createFilterbank(
            sr: spec.sampleRate, nFFT: self.nFFT, nMel: self.nMel, nFreq: self.nFreq)
    }

    /// MLX log-mel, shape [1, T, features] float32 (same as the reference).
    func computeMLX(audio: [Float]) -> MLXArray? {
        let x = MLXArray(audio) // [N]
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

        // reflect pad (same formula as the reference _pad)
        let pad = self.nFFT / 2
        let prefixRev = Array(y[1..<(pad + 1)].asArray(Float.self).reversed())
        let suffixRev = Array(y[(y.size - 1 - pad)..<(y.size - 1)].asArray(Float.self).reversed())
        let padded = concatenated([MLXArray(prefixRev), y, MLXArray(suffixRev)])

        // STFT frames (same indexing as the reference as_strided)
        let t = (padded.size - self.winLength + self.hop) / self.hop
        guard t > 0 else { return nil }
        let win = self.window[0..<self.winLength]
        var frames = [MLXArray]()
        frames.reserveCapacity(t)
        for i in 0..<t {
            let start = i * self.hop
            let frame = padded[start..<(start + self.winLength)] * win
            frames.append(frame)
        }
        let framed = stacked(frames, axis: 0) // [T, winLength]
        let framedFull = MLX.padded(
            framed,
            widths: [IntOrPair((0, 0)), IntOrPair((0, self.nFFT - self.winLength))])
        let specArr = rfft(framedFull, n: self.nFFT, axis: 1) // [T, 257] complex64
        // Split complex into real/imag via strided view: [T, 514] = [re, im] pairs.
        let view = specArr.view(dtype: .float32) // [T, 514]
        let rePart = view[0..., .stride(from: 0, to: 2 * self.nFreq, by: 2)] // [T, nFreq] reals
        let imPart = view[0..., .stride(from: 1, to: 2 * self.nFreq, by: 2)] // [T, nFreq] imags
        let mag = abs(rePart) + abs(imPart) // [T, nFreq]
        let magPow = self.magPower == 1.0 ? mag : pow(mag, self.magPower) // [T, nFreq]

        // mel: [nMel, nFreq] @ [nFreq, T] -> [nMel, T]
        let melSpec = matmul(self.filterbank, magPow.T) // [nMel, T]
        let logSpec = log(melSpec + 1e-5) // [nMel, T]

        // per_feature: mean/std over frames (axis 1), per mel bin
        let meanV = mean(logSpec, axis: 1, keepDims: true) // [nMel, 1]
        let d = logSpec - meanV
        let std = sqrt(mean(d * d, axis: 1, keepDims: true))
        let normalized = (logSpec - meanV) / (std + 1e-5) // [nMel, T]

        return normalized.T.expandedDimensions(axis: 0) // [1, T, nMel]
    }

    // MARK: - librosa slaney mel filterbank (matches librosa.filters.mel)

    private static func createFilterbank(sr: Int, nFFT: Int, nMel: Int, nFreq: Int) -> MLXArray {
        let fMax = Double(sr) / 2.0

        // librosa hz_to_mel / mel_to_hz with htk=false (Slaney piecewise).
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
                // librosa slaney normalization: constant energy per channel
                let width = fRight - fLeft
                if width > 0 { v *= 2.0 / width }
                flat[i * nFreq + j] = Float(v)
            }
        }
        return MLXArray(flat, [nMel, nFreq])
    }
}
