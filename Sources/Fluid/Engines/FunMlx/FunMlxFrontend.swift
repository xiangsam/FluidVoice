import Foundation
import MLX

// Fun-ASR (MLT) Nano frontend, mirroring mlx-audio-plus
// `stt.models.funasr.audio` exactly:
// - Hamming(400) window, reflect-padded centered STFT (hop 160)
// - power spectrum |rfft|^2 over 201 bins, drop last frame
// - mel filterbank: HTK mel scale + slaney normalization (f_min 0, f_max nyquist)
// - log(max(mel, 1e-10))
// - LFR m=7 n=6 (stack + subsample)
// - per-utterance CMVN: (x - mean) / (std + 1e-6)

final class FunMlxFrontend {

    private let spec: FunMlxSpec
    private let winLen: Int
    private let winInc: Int
    private let nFreq: Int // winLen / 2 + 1
    private let nMel: Int

    private let hamming: MLXArray
    private let melBanks: MLXArray // [nMel, nFreq] (htk scale, slaney norm)

    init(spec: FunMlxSpec) {
        self.spec = spec
        self.winLen = spec.winLen
        self.winInc = spec.winInc
        self.nFreq = spec.winLen / 2 + 1
        self.nMel = spec.nMel

        // Hamming (symmetric): 0.54 - 0.46 cos(2 pi n / (N-1))
        let denom = Float(self.winLen - 1)
        self.hamming = MLXArray(
            (0..<self.winLen).map { i -> Float in
                0.54 - 0.46 * cosf(2.0 * .pi * Float(i) / denom)
            })

        self.melBanks = Self.createFilterbank(
            sr: spec.fs, nFFT: self.winLen, nMel: self.nMel, nFreq: self.nFreq)
    }

    /// Returns (feats [1, T, nMel*lfrM], speech_length). Mirrors
    /// `preprocess_audio(audio)` in mlx-audio-plus.
    func compute(audio: [Float]) -> (MLXArray, Int)? {
        let n = audio.count
        let pad = self.winLen / 2
        guard n >= 2 * pad + 2 else { return nil }

        let x = MLXArray(audio)
        // reflect pad (stft center=True, pad_mode="reflect")
        let prefixRev = Array(x[1..<(pad + 1)].asArray(Float.self).reversed())
        let suffixRev = Array(x[(n - 1 - pad)..<(n - 1)].asArray(Float.self).reversed())
        let padded = concatenated([MLXArray(prefixRev), x, MLXArray(suffixRev)])

        let numFrames = 1 + (padded.size - self.winLen) / self.winInc // 1 + n/winInc
        guard numFrames > 1 else { return nil }

        var frames = [MLXArray]()
        frames.reserveCapacity(numFrames)
        for i in 0..<numFrames {
            let start = i * self.winInc
            let frame = padded[start..<(start + self.winLen)] * self.hamming
            frames.append(frame)
        }
        let strided = stacked(frames, axis: 0) // [numFrames, winLen]
        let specArr = rfft(strided, n: self.winLen, axis: 1) // [numFrames, nFreq] complex
        let mag = abs(specArr)
        let power = mag * mag
        let magnitudes = power[0..<(numFrames - 1), 0...] // drop last frame

        let melSpec = matmul(magnitudes, self.melBanks.T) // [F, nMel]
        let logMel = log(maximum(melSpec, 1e-10)) // [F, nMel]

        // LFR (m/n) - same framing as mlx-audio-plus apply_lfr
        let lfrM = self.spec.lfrM
        let lfrN = self.spec.lfrN
        let t = logMel.shape[0]
        let tLfr = (t + lfrN - 1) / lfrN
        let leftPad = (lfrM - 1) / 2

        var paddedFeats = logMel
        if leftPad > 0 {
            let first = logMel[0..<1]
            let reps = MLXArray.repeated(first, count: leftPad, axis: 0)
            paddedFeats = concatenated([reps, logMel], axis: 0)
        }
        let paddedT = paddedFeats.shape[0]
        let d = self.nMel

        var outFrames = [MLXArray]()
        outFrames.reserveCapacity(tLfr)
        for i in 0..<tLfr {
            let start = i * lfrN
            let end = start + lfrM
            var frame: MLXArray
            if end <= paddedT {
                frame = paddedFeats[start..<end].reshaped([lfrM * d])
            } else {
                let avail = paddedFeats[start..<paddedT]
                let padCount = end - paddedT
                let last = paddedFeats[(paddedT - 1)..<paddedT]
                let reps = MLXArray.repeated(last, count: padCount, axis: 0)
                frame = concatenated([avail, reps], axis: 0).reshaped([lfrM * d])
            }
            outFrames.append(frame)
        }
        let feats = stacked(outFrames, axis: 0) // [T_lfr, m*D]

        // Per-utterance CMVN
        let featsMean = mean(feats, axis: 0, keepDims: true)
        let featsStd = std(feats, axis: 0, keepDims: true) + 1e-6
        let normed = (feats - featsMean) / featsStd

        return (normed[.newAxis, 0..., 0...], tLfr)
    }

    // MARK: - HTK mel scale + slaney normalization (librosa filters.mel,
    // matching mlx_audio.dsp.mel_filters(sr, n_fft, n_mels, norm="slaney",
    // mel_scale="htk") with f_min=0, f_max=sr/2).

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
