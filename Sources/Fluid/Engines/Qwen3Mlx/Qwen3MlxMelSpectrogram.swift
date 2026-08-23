import Accelerate
import Foundation

// Mel spectrogram frontend for Qwen3-ASR (MLX engine).
//
// Mirrors `qwen3_asr_mlx/audio.py::log_mel_spectrogram` exactly:
// - n_fft: 400, hop_length: 160, n_mels: 128, F_MIN 0, F_MAX 8000
// - Symmetric Hann window (np.hanning(400))
// - numpy "reflect" padding of n_fft/2 on both sides
// - n_frames = 1 + (len(audio)) / hop, then the LAST STFT frame is dropped
// - Power spectrum |X|^2
// - Slaney mel filterbank normalised by full filter width (1/width)
// - log10(max(mel, 1e-10)), then max(x, x.max() - 8), then (x + 4) / 4
//
// The DFT is computed via a pre-computed cos/sin matrix (vDSP_mmul) because
// vDSP's radix-2 FFT cannot handle n_fft = 400 directly.
final class Qwen3MlxMelSpectrogram {

    private let nFFT: Int = 400
    private let hopLength: Int = 160
    private let nMels: Int = 128
    private let sampleRate: Int = 16_000

    private var numFreqBins: Int { nFFT / 2 + 1 }

    private let hannWindow: [Float]
    private let melFilterbankFlat: [Float]  // [nMels * numFreqBins] row-major
    private let dftCos: [Float]
    private let dftSin: [Float]

    private var windowedFrame: [Float]
    private var realPart: [Float]
    private var imagPart: [Float]
    private var powerSpec: [Float]
    private var imagSq: [Float]
    private var melFrame: [Float]

    init() {
        let numFreqBins = nFFT / 2 + 1

        // Symmetric Hann: np.hanning(400) => 0.5 * (1 - cos(2*pi*n/(N-1)))
        var window = [Float](repeating: 0, count: nFFT)
        let denom = Float(nFFT - 1)
        for i in 0..<nFFT {
            window[i] = 0.5 * (1.0 - cosf(2.0 * .pi * Float(i) / denom))
        }
        self.hannWindow = window

        // Slaney mel filterbank normalised by full filter width (Hz).
        let filterbank = Self.createMelFilterbank(
            nFFT: nFFT,
            nMels: nMels,
            sampleRate: sampleRate,
            fMin: 0.0,
            fMax: 8000.0
        )

        var flat = [Float](repeating: 0, count: nMels * numFreqBins)
        for m in 0..<nMels {
            for f in 0..<numFreqBins {
                flat[m * numFreqBins + f] = filterbank[m][f]
            }
        }
        self.melFilterbankFlat = flat

        // Pre-compute DFT cos/sin tables for k=0..numFreqBins-1, n=0..nFFT-1
        var cosTable = [Float](repeating: 0, count: numFreqBins * nFFT)
        var sinTable = [Float](repeating: 0, count: numFreqBins * nFFT)
        let invN = Float(2.0 * .pi) / Float(nFFT)
        for k in 0..<numFreqBins {
            for n in 0..<nFFT {
                let angle = -invN * Float(k) * Float(n)
                cosTable[k * nFFT + n] = cosf(angle)
                sinTable[k * nFFT + n] = sinf(angle)
            }
        }
        self.dftCos = cosTable
        self.dftSin = sinTable

        self.windowedFrame = [Float](repeating: 0, count: nFFT)
        self.realPart = [Float](repeating: 0, count: numFreqBins)
        self.imagPart = [Float](repeating: 0, count: numFreqBins)
        self.powerSpec = [Float](repeating: 0, count: numFreqBins)
        self.imagSq = [Float](repeating: 0, count: numFreqBins)
        self.melFrame = [Float](repeating: 0, count: nMels)
    }

    /// Returns mel spectrogram as [nMels][nFrames] with
    /// nFrames = audio.count / hopLength (last STFT frame dropped).
    func compute(audio: [Float]) -> [[Float]] {
        let numFrames = audio.count / hopLength
        guard numFrames > 0 else { return [] }
        let numFreqBins = self.numFreqBins
        let n = audio.count
        let pad = nFFT / 2

        // numpy-style reflect padding: [a[pad-1], ..., a[0], a[0..<n], a[n-2], ..., a[n-1-pad]]
        var padded = [Float](repeating: 0, count: n + 2 * pad)
        for i in 0..<pad {
            let src = min(pad - 1 - i, n - 1)
            padded[i] = audio[src]
        }
        for i in 0..<n {
            padded[pad + i] = audio[i]
        }
        for i in 0..<pad {
            let src = max(n - 2 - i, 0)
            padded[pad + n + i] = audio[src]
        }

        var mel = [[Float]](
            repeating: [Float](repeating: 0, count: numFrames),
            count: nMels
        )

        for frameIdx in 0..<numFrames {
            let startIdx = frameIdx * hopLength

            // Frame + window from reflect-padded signal
            for i in 0..<nFFT {
                windowedFrame[i] = padded[startIdx + i] * hannWindow[i]
            }

            dftCos.withUnsafeBufferPointer { cosPtr in
                windowedFrame.withUnsafeBufferPointer { xPtr in
                    realPart.withUnsafeMutableBufferPointer { outPtr in
                        vDSP_mmul(
                            cosPtr.baseAddress!, 1,
                            xPtr.baseAddress!, 1,
                            outPtr.baseAddress!, 1,
                            vDSP_Length(numFreqBins),
                            vDSP_Length(1),
                            vDSP_Length(nFFT)
                        )
                    }
                }
            }

            dftSin.withUnsafeBufferPointer { sinPtr in
                windowedFrame.withUnsafeBufferPointer { xPtr in
                    imagPart.withUnsafeMutableBufferPointer { outPtr in
                        vDSP_mmul(
                            sinPtr.baseAddress!, 1,
                            xPtr.baseAddress!, 1,
                            outPtr.baseAddress!, 1,
                            vDSP_Length(numFreqBins),
                            vDSP_Length(1),
                            vDSP_Length(nFFT)
                        )
                    }
                }
            }

            // Power spectrum
            vDSP_vsq(realPart, 1, &powerSpec, 1, vDSP_Length(numFreqBins))
            vDSP_vsq(imagPart, 1, &imagSq, 1, vDSP_Length(numFreqBins))
            vDSP_vadd(powerSpec, 1, imagSq, 1, &powerSpec, 1, vDSP_Length(numFreqBins))

            // Mel filterbank
            melFilterbankFlat.withUnsafeBufferPointer { filterPtr in
                powerSpec.withUnsafeBufferPointer { specPtr in
                    melFrame.withUnsafeMutableBufferPointer { outPtr in
                        vDSP_mmul(
                            filterPtr.baseAddress!, 1,
                            specPtr.baseAddress!, 1,
                            outPtr.baseAddress!, 1,
                            vDSP_Length(nMels),
                            vDSP_Length(1),
                            vDSP_Length(numFreqBins)
                        )
                    }
                }
            }

            var minClip: Float = 1e-10
            var maxClip: Float = Float.greatestFiniteMagnitude
            vDSP_vclip(melFrame, 1, &minClip, &maxClip, &melFrame, 1, vDSP_Length(nMels))

            var count = Int32(nMels)
            vvlog10f(&melFrame, melFrame, &count)

            for melIdx in 0..<nMels {
                mel[melIdx][frameIdx] = melFrame[melIdx]
            }
        }

        // Dynamic range: max(x, globalMax - 8.0)
        var globalMax: Float = -Float.infinity
        for melIdx in 0..<nMels {
            var rowMax: Float = 0
            vDSP_maxv(mel[melIdx], 1, &rowMax, vDSP_Length(numFrames))
            globalMax = max(globalMax, rowMax)
        }
        var minVal = globalMax - 8.0
        var high = Float.greatestFiniteMagnitude
        for melIdx in 0..<nMels {
            mel[melIdx].withUnsafeMutableBufferPointer { buffer in
                vDSP_vclip(buffer.baseAddress!, 1, &minVal, &high, buffer.baseAddress!, 1, vDSP_Length(numFrames))
            }
        }

        // (x + 4) / 4
        var addVal: Float = 4.0
        var divVal: Float = 4.0
        for melIdx in 0..<nMels {
            mel[melIdx].withUnsafeMutableBufferPointer { buffer in
                vDSP_vsadd(buffer.baseAddress!, 1, &addVal, buffer.baseAddress!, 1, vDSP_Length(numFrames))
                vDSP_vsdiv(buffer.baseAddress!, 1, &divVal, buffer.baseAddress!, 1, vDSP_Length(numFrames))
            }
        }

        return mel
    }

    // MARK: Private - Mel Filterbank (Slaney, 1/width normalisation)

    private static func createMelFilterbank(
        nFFT: Int,
        nMels: Int,
        sampleRate: Int,
        fMin: Float,
        fMax: Float
    ) -> [[Float]] {
        let numFreqBins = nFFT / 2 + 1

        var fftFreqs = [Float](repeating: 0, count: numFreqBins)
        for i in 0..<numFreqBins {
            fftFreqs[i] = Float(i) * Float(sampleRate) / Float(nFFT)
        }

        func hzToMel(_ hz: Float) -> Float {
            2595.0 * log10f(1.0 + hz / 700.0)
        }
        func melToHz(_ mel: Float) -> Float {
            700.0 * (powf(10.0, mel / 2595.0) - 1.0)
        }

        let melMin = hzToMel(fMin)
        let melMax = hzToMel(fMax)
        var melPoints = [Float](repeating: 0, count: nMels + 2)
        for i in 0..<(nMels + 2) {
            let mel = melMin + Float(i) * (melMax - melMin) / Float(nMels + 1)
            melPoints[i] = melToHz(mel)
        }

        var filterbank = [[Float]](
            repeating: [Float](repeating: 0, count: numFreqBins),
            count: nMels
        )

        for i in 0..<nMels {
            let fLeft = melPoints[i]
            let fCenter = melPoints[i + 1]
            let fRight = melPoints[i + 2]
            let width = fRight - fLeft
            for f in 0..<numFreqBins {
                let up = (fftFreqs[f] - fLeft) / (fCenter - fLeft)
                let down = (fRight - fftFreqs[f]) / (fRight - fCenter)
                var v = max(0, min(up, down))
                if width > 0 { v /= width }
                filterbank[i][f] = v
            }
        }

        return filterbank
    }
}
