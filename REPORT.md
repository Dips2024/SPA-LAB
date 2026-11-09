# Mini-project report

**Title:** Intelligent Audio Noise Removal and Signal Quality Analyzer  
**Course:** EC6675 Signal Processing Application Laboratory, NIT Rourkela  
**Foundation:** Module 2 (noise addition, moving average) and Module 3 (Butterworth and FIR filters)

## Objective and extension beyond the lab exercises

The lab techniques are combined into one local application that accepts a WAV file or produces a repeatable test signal, compares three core filters under chosen settings, shows time, FFT, and spectrogram views, explains an evidence-based recommendation, and exports the result. A combined notch-and-FIR option appears when narrow 50/60 Hz hum is detected. A noise-profile method can use a selected noise-only interval. An offline machine-learning panel recognizes noise patterns as an extra diagnostic. The project adds user interaction, validation, quantitative comparison, and an honest distinction between known-reference and unknown-reference audio.

## Signal flow

`WAV or synthetic clean + noise → sample validation → core filters (+ hum notch/FIR and optional noise profile) → quality table → recommendation → plots/audio/export`

For the synthetic case, the clean signal contains 220, 440, and 880 Hz components with a slow amplitude envelope. A fixed random seed makes noise reproducible. Options are high-frequency hiss, white noise, and 50 Hz hum plus hiss. If its peak would clip, clean and noisy signals receive the **same scale**.

## Methods

1. **Moving average:** For odd window length `M`, output at sample `n` is the mean of nearby `M` samples. It is a centered finite impulse response smoother with reflected boundaries. Its cutoff is determined by `M` and differs from the Butterworth/FIR cutoff setting.
2. **Butterworth:** Fourth-order low-pass IIR, implemented as second-order sections for numerical stability. Forward/backward application removes phase delay in this offline project. The combined response is steeper than the one-pass filter.
3. **FIR:** Hamming-windowed low-pass with user-selected odd tap count. Forward/backward application removes phase delay. More taps narrow the transition but increase work. The chosen tap count is a design parameter, not proof of optimality.
4. **Adaptive hum option:** A narrow 50 or 60 Hz peak is detected from the input FFT using peak prominence and energy checks. When found, a Q=30 IIR notch removes that tone, followed by the FIR low-pass for hiss. A useful bass note near mains frequency can cause a false detection.
5. **Noise profile:** An uploaded recording's chosen noise-only segment supplies the median magnitude spectrum of short-time Fourier transform frames. A smoothed mask reduces matching spectral energy throughout the recording, with a 0.12 floor to avoid hard spectral holes. The synthetic case uses a 0.5-second sample of its generated noise. A segment containing useful audio can cause audible distortion.
6. **FFT and spectrogram:** A Hann-windowed real FFT of up to the first 65,536 samples shows overall frequency content; a spectrogram shows changes over time. Before/after spectrograms share a color scale. On stereo, plotted waveform and spectrogram use the left channel, while FFT magnitudes are averaged across channels.

## Evaluation and recommendation

When a clean reference `s[n]` exists, **MSE** = mean of `(s[n] − y[n])²`, and **SNR** = `10 log10(mean(s[n]²) / MSE)`. They are calculated on floating-point outputs before 16-bit export. The synthetic recommendation is the filter with the smallest MSE, which also has the highest SNR for the same reference. The unfiltered noisy input is shown as a baseline. The app explicitly warns if even the best filter is worse than the baseline.

For uploaded audio, the clean signal is unknown, so true MSE and SNR cannot be measured. Welch power spectra provide conditional **proxies**: reduction above `1.2 × cutoff`, power retained below `0.8 × cutoff`, and change in the lower-band spectral magnitudes. When hum is detected, a separate hum-reduction measure is added and the narrow hum region is excluded from the preservation measure. The transition region is ignored. Reported reductions are capped at 80 dB. Among filters with at most 10% lower-band spectral change, the rule selects the greatest measured high-band plus hum reduction; if none qualify, it selects the least-changing option. If the upper band is nearly empty, it also selects the least-changing option and notes that filtering may be unnecessary.

This rule assumes high-frequency energy is unwanted noise. It can fail for cymbals, consonants, and other useful high-frequency audio. The displayed measurements do **not** estimate perceptual quality, true SNR, or MSE. Listen before accepting the output.

## Offline AI assistant

The app generates 48 labeled examples of high-frequency hiss, white noise, and 50 Hz hum plus hiss at four noise levels. For each example, it measures six normalized spectral-band powers. A 5-nearest-neighbors classifier compares the input with these examples and displays the most similar noise pattern. The displayed match percentage is a weighted vote share, **not a calibrated probability**. No cloud API, external model, or network request is used.

This model recognizes patterns similar to its generated training set. Real speech, music, or different noises can confuse it. It does not replace the reference MSE/SNR calculation or the proxy-based filter comparison. Actual notch activation uses a separate narrow-peak detector, not the AI pattern guess.

## Expected observations and limitations

- High-frequency hiss should become weaker after low-pass filtering. A lower cutoff or longer moving-average window usually removes more high-frequency content but can dull useful sound.
- White noise overlaps the clean signal's band, so no low-pass can remove all of it without losing signal.
- A 50 Hz hum lies below the clean tones. The three low-pass filters generally leave it in place; the added notch/FIR combination handles a detected narrow hum tone. Broadband low-frequency noise or a useful tone near 50/60 Hz remains difficult to distinguish automatically.
- Noise-profile removal requires a representative noise-only excerpt. It may create artifacts or remove wanted sound if that excerpt includes speech or music.
- Forward/backward filters require the whole file and are not a live/real-time design. Recordings longer than 20 seconds must be trimmed before upload.
- Filter edge behavior and 16-bit export can introduce small differences; the report's metrics describe the float processing result.

## Viva questions and short answers

**Why use a clean simulated signal?** It gives a ground truth for genuine MSE and SNR comparisons.  
**Why these filters?** Moving average is a simple Module 2 baseline; Butterworth IIR and window-designed FIR extend Module 3. The detected-hum option adds a notch followed by FIR because low-pass filtering alone misses narrow 50/60 Hz interference. A moving average is itself an FIR filter, though simpler than the separately designed FIR low-pass.  
**Why no SNR for uploaded recordings?** We do not know the desired clean waveform or exact noise error.  
**Difference between IIR and FIR?** Butterworth IIR uses feedback and relatively few coefficients; FIR uses a finite weighted sum and can have linear phase in one pass.  
**Why FFT?** It reveals frequency components and shows attenuation across the low-pass cutoff.  
**Why add a notch?** It solves a failure case of the three low-pass filters: a narrow 50/60 Hz hum below the useful signal band.  
**Why add a noise profile?** A measured example of the recording's noise helps suppress noise that a simple low-pass cannot isolate; its quality depends on choosing a truly noise-only segment.  
**Can this remove any noise?** No. Noise overlapping useful frequencies may remain.  
**Why forward/backward filtering?** It cancels phase delay for offline playback, at the cost of using the whole recording and changing the effective magnitude response.

## References

- [SciPy Butterworth design](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html)
- [SciPy second-order-section forward/backward filtering](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfiltfilt.html)
- [SciPy FIR window design](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.firwin.html)
- [SciPy Welch spectrum](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html)
