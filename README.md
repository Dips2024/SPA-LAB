# Intelligent Audio Noise Removal and Signal Quality Analyzer

**NIT Rourkela EC6675 Signal Processing Application Laboratory — mini-project based on Modules 2 and 3.** The app runs on one laptop. It needs no microphone, sensor, cloud account, or internet after its Python packages are installed.

## Run locally

Use Python **3.11–3.14**. In a terminal opened inside this folder:

### Windows PowerShell

On this laptop, Python 3.14 is installed at the path used below. First extract the ZIP, open PowerShell **inside the `intelligent-audio-analyzer` folder**, then run:

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python314\python.exe" -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The install step needs internet once. For later launches, open PowerShell in the same folder and run only the last command. On a different Windows computer, use its Python 3.11–3.14 executable for the first command; `py -3.14 -m venv .venv` also works when the Python launcher knows that installation. The commands use the environment's Python directly, so PowerShell activation is unnecessary.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Streamlit prints a local address, usually `http://localhost:8501`. Open it in your browser. Stop with **Ctrl+C**. Install the requirements while connected to the internet once; afterward the app processes data locally and makes no external service calls. To make sample WAVs again, run `python create_samples.py`.

## Hosted version

The same source can run on Streamlit Community Cloud from the `main` branch with `app.py` as the entry point and `requirements.txt` in the repository root. Choose Python 3.14 if the deployment form asks for a version. A hosted copy processes uploaded WAV files on the hosting server; use the local version for recordings you do not want to upload.

## Five-minute demonstration

1. Keep **Simulated signal**, **High-frequency hiss**, and the initial settings. The **AI noise pattern** panel should identify high-frequency hiss. Play the original and output in **Listen & export**.
2. Compare the SNR and MSE table. The recommendation minimizes MSE against the **known clean synthetic signal**. Switch between the three outputs and inspect **Signal analysis**.
3. Change noise to **50 Hz hum + hiss**. The app detects the narrow 50 Hz tone and adds **Notch + FIR (50 Hz)** to the comparison. Explain why the three low-pass filters alone leave much of the hum.
4. Select **Upload WAV** and upload `sample_data/noisy_demo.wav` or your own recording. The app now displays **proxies only** because it no longer knows the clean waveform, even if you uploaded the sample file.
5. Turn on **Compare noise-profile removal**. Mark a short interval that contains only background noise, and listen to its preview. The **Noise profile** row appears in the comparison. If the interval contains speech or music, select another interval.
6. Open **Signal analysis** to compare the waveform, FFT, and before/after spectrogram. Play and download a chosen filtered WAV.

The AI panel uses a small **5-nearest-neighbors classifier** trained locally at startup on 48 labeled synthetic signals. It compares six spectral features and reports a template-vote share, not a calibrated probability. Its pattern guess is advisory. The filter recommendation still uses the measured reference result or the stated no-reference proxy rule.

## Input limits and behavior

- Uncompressed RIFF/WAV: 8/16/24/32-bit PCM or 32/64-bit IEEE float, mono or stereo.
- Sample rate 8–48 kHz, duration 0.25–20 seconds, file at most 25 MB.
- Invalid, unsupported, non-finite, or out-of-range audio produces a clear error. No silent resampling or channel mixing occurs.
- Processing uses floating-point samples. Playback and export convert to 16-bit PCM, clipping only if a filtered sample exceeds ±1; the app warns in that case.
- The cut-off slider controls Butterworth and FIR. The moving average has its own window length and a different response.
- A narrow **Notch + FIR** option appears automatically when a prominent 50 or 60 Hz tone is found. The detector can mistake a useful musical bass tone for hum, so listen before exporting.
- **Noise profile** estimates a noise spectrum from a selected interval and applies a smoothed short-time Fourier transform mask. For the simulated signal, the app provides a separate 0.5-second noise-only sample. A recording without a noise-only interval is a poor fit for this method.
- The selected recording name is shown in a prominent card above the results. The spectrograms use one shared color scale so visual differences are comparable.

## Project files

| File | Purpose |
|---|---|
| `app.py` | Local Streamlit dashboard, controls, plots, playback, download |
| `dsp.py` | WAV validation, synthetic signal, filters, metrics, recommendation |
| `ai.py` | Offline noise-pattern classifier and its generated training bank |
| `create_samples.py` | Rebuild the included demo WAVs |
| `sample_data/` | Noisy and clean 2-second demonstration WAVs |
| `tests/test_dsp.py` | Core behavior and validation tests |
| `REPORT.md` | Method, results interpretation, limitations, viva questions |

Run core checks with `python -m unittest discover -s tests -v`.

## Dependencies

Streamlit provides the local browser interface; NumPy handles arrays/FFT and the small classifier; SciPy reads WAV and designs/applies filters; Matplotlib draws plots. The app has no network API or model download.
