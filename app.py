"""Local Streamlit interface for the EC6675 audio filtering mini-project."""

from __future__ import annotations

from html import escape
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from scipy import signal

from ai import predict_pattern, training_bank
from dsp import (
    MAX_UPLOAD_BYTES,
    AudioData,
    apply_filters,
    generate_test_signal,
    proxy_scores,
    read_wav,
    recommend_with_reference,
    reference_scores,
    spectral_gate,
    wav_bytes,
)


st.set_page_config(page_title="Audio Noise Lab | EC6675", page_icon="🎛️", layout="wide")
st.markdown(
    """
    <style>
      .block-container {max-width: 1180px; padding-top: 1.2rem;}
      h1,h2,h3 {color: #163344;}
      div[data-testid="stMetric"] {background: #eaf3f4; border-radius: 12px; padding: 12px 16px;}
      .eyebrow {color:#a7f3d0; font-weight:700; letter-spacing:.12em; font-size:.72rem;}
      .hero {background:linear-gradient(115deg,#163344,#0d676c);color:white;padding:1rem 1.5rem;border-radius:16px;margin-bottom:1rem;}
      .hero h1 {color:white;margin:0.15rem 0;font-size:2rem;}
      .source-card {background:#fff;border:2px solid #0d9488;border-radius:12px;padding:.7rem 1rem;margin:.4rem 0 1rem;color:#163344;}
      .source-card span {font-size:.72rem;font-weight:700;letter-spacing:.08em;color:#0d676c;}
      .source-card strong {display:block;font-size:1.12rem;overflow-wrap:anywhere;}
    </style>
    <div class="hero"><div class="eyebrow">NIT ROURKELA · EC6675 · MODULES 2 + 3</div>
    <h1>Audio Noise Lab</h1></div>
    """,
    unsafe_allow_html=True,
)


def waveform_figure(original: np.ndarray, cleaned: np.ndarray, fs: int, noise_range: tuple[float, float] | None = None) -> plt.Figure:
    count = len(original)
    step = max(1, count // 5500)
    time = np.arange(0, count, step) / fs
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.2), sharex=True, sharey=True, constrained_layout=True)
    for ax, y, title, color in zip(
        axes,
        (original, cleaned),
        ("Original / noisy", "Selected filter output"),
        ("#d97706", "#0d9488"),
    ):
        ax.plot(time, y[::step, 0], color=color, linewidth=0.8)
        ax.set_title(title, loc="left", fontsize=11, weight="bold")
        ax.set_ylabel("Amplitude")
        ax.set_ylim(-1.05, 1.05)
        ax.grid(alpha=0.2)
        if noise_range is not None:
            ax.axvspan(*noise_range, color="#99f6e4", alpha=0.45)
    axes[-1].set_xlabel("Time (seconds)")
    return fig


def fft_curve(x: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(x), 65536)
    part = x[:n]
    window = np.hanning(n)[:, None]
    magnitude = np.mean(np.abs(np.fft.rfft(part * window, axis=0)), axis=1)
    amplitude = 2 * magnitude / max(np.sum(window), 1)
    freq = np.fft.rfftfreq(n, 1 / fs)
    return freq, 20 * np.log10(np.maximum(amplitude, 1e-7))


def spectrum_figure(original: np.ndarray, cleaned: np.ndarray, fs: int, cutoff: int) -> plt.Figure:
    f0, db0 = fft_curve(original, fs)
    f1, db1 = fft_curve(cleaned, fs)
    fig, ax = plt.subplots(figsize=(11, 4.2), constrained_layout=True)
    ax.plot(f0, db0, color="#d97706", linewidth=1, alpha=0.8, label="Original / noisy")
    ax.plot(f1, db1, color="#0d9488", linewidth=1, alpha=0.85, label="Selected output")
    ax.axvline(cutoff, color="#64748b", linestyle="--", linewidth=1, label="IIR/FIR cutoff")
    ax.set_xlim(0, min(6000, fs / 2))
    ax.set_ylim(-115, 5)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("FFT magnitude (dBFS approx.)")
    ax.set_title("Frequency spectrum", loc="left", fontsize=11, weight="bold")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", fontsize=9)
    return fig


def spectrogram_figure(original: np.ndarray, cleaned: np.ndarray, fs: int) -> plt.Figure:
    frame = min(2048, max(512, fs // 8))
    spectra = []
    for audio in (original, cleaned):
        f, t, magnitude = signal.spectrogram(audio[:, 0], fs=fs, nperseg=frame, noverlap=3 * frame // 4, mode="magnitude")
        spectra.append((f, t, magnitude))
    peak = max(float(np.max(item[2])) for item in spectra)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, sharey=True, constrained_layout=True)
    for ax, (f, t, magnitude), title in zip(axes, spectra, ("Original / noisy", "Selected output")):
        db = 20 * np.log10(np.maximum(magnitude, peak * 1e-4) / max(peak, 1e-12))
        mesh = ax.pcolormesh(t, f, db, shading="auto", cmap="magma", vmin=-80, vmax=0)
        ax.set_ylim(0, min(6000, fs / 2))
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(title, loc="left", fontsize=11, weight="bold")
    axes[-1].set_xlabel("Time (seconds)")
    fig.colorbar(mesh, ax=axes, label="Level relative to input peak (dB)")
    return fig


def summary_table(rows: list[dict]) -> None:
    st.dataframe(rows, width="stretch", hide_index=True)


@st.cache_resource
def get_ai_bank() -> tuple[np.ndarray, np.ndarray]:
    return training_bank()


with st.sidebar:
    st.header("Source")
    source = st.radio("Audio source", ["Simulated signal", "Upload WAV"], horizontal=False)
    data: AudioData | None = None
    if source == "Upload WAV":
        upload = st.file_uploader("Choose an uncompressed WAV", type=["wav"], help="Mono/stereo, 8–48 kHz, 0.25–20 s, at most 25 MB. On a hosted copy, the file is processed on the hosting server.")
        if upload is not None:
            try:
                raw = upload.getvalue()
                if len(raw) > MAX_UPLOAD_BYTES:
                    raise ValueError("WAV exceeds the 25 MB limit.")
                data = read_wav(raw, upload.name)
            except ValueError as exc:
                st.error(str(exc))
    else:
        kind = st.selectbox("Noise type", ["High-frequency hiss", "White noise", "50 Hz hum + hiss"])
        level = st.slider("Noise RMS amplitude", 0.05, 0.40, 0.22, 0.01)
        with st.expander("Reproducibility"):
            seed = st.number_input("Random seed", 0, 10000, 7, 1)
        data = generate_test_signal(noise_level=level, noise_kind=kind, seed=int(seed))
    st.divider()
    st.header("Filter settings")
    if data is None:
        st.caption("Select a valid WAV to enable processing.")
        st.stop()
    max_cutoff = min(6000, 100 * (int(0.45 * data.sample_rate - 1) // 100))
    cutoff = st.slider("Low-pass cutoff (Hz)", 200, max_cutoff, min(1400, max_cutoff), 100)
    with st.expander("Advanced settings"):
        window = st.slider("Moving-average samples (odd)", 3, 31, 7, 2)
        taps = st.slider("FIR taps (odd)", 31, 401, 151, 10)
    st.divider()
    st.header("Noise profile")
    use_profile = st.toggle("Compare noise-profile removal", value=source == "Simulated signal")
    noise_excerpt = None
    profile_range = None
    if use_profile and data.clean_reference is not None:
        noise_excerpt = (data.samples - data.clean_reference)[: int(0.5 * data.sample_rate)]
        st.caption("Using a 0.5-second generated noise-only sample.")
    elif use_profile:
        profile_range = st.slider("Select a noise-only interval (seconds)", min_value=0.0, max_value=round(data.duration, 2), value=(0.0, round(min(0.5, data.duration), 2)), step=0.01)
        start, end = (int(round(v * data.sample_rate)) for v in profile_range)
        if end - start >= int(0.1 * data.sample_rate):
            noise_excerpt = data.samples[start:end]
            st.audio(wav_bytes(noise_excerpt, data.sample_rate), format="audio/wav")
        else:
            st.warning("Select at least 0.1 seconds containing only noise.")
    strength = st.slider("Noise-profile strength", 0.5, 1.5, 1.0, 0.1) if use_profile else 1.0

try:
    outputs = apply_filters(data, float(cutoff), window, taps)
    if noise_excerpt is not None:
        outputs["Noise profile"] = spectral_gate(data, noise_excerpt, strength)
    if data.clean_reference is not None:
        rows = reference_scores(data, outputs)
        recommended, reason = recommend_with_reference(rows)
        is_reference = True
    else:
        rows, recommended, reason = proxy_scores(data, outputs, float(cutoff))
        is_reference = False
except (ValueError, MemoryError) as exc:
    st.error(f"Processing could not finish: {exc}")
    st.stop()

st.markdown(f'<div class="source-card"><span>CURRENT RECORDING</span><strong>{escape(data.name)}</strong></div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
col1.metric("Duration", f"{data.duration:.2f} s")
col2.metric("Sample rate", f"{data.sample_rate:,} Hz")
col3.metric("Channels", "Mono" if data.samples.shape[1] == 1 else "Stereo")

pattern, match = predict_pattern(data, get_ai_bank())
if pattern == "50 Hz hum + hiss":
    guidance = "A notch option appears when a strong 50/60 Hz tone is detected."
else:
    guidance = "Compare the measured filter results below."
st.info(f"**AI noise pattern:** {pattern} · {match:.0%} template match. {guidance}")
with st.expander("About the AI result"):
    st.caption("An offline 5-nearest-neighbors model compares six spectral features with 48 generated, labeled examples. The match is a vote share, not a calibrated probability. Music or speech may resemble a noise pattern; the quality table determines the filter suggestion.")

st.subheader("Compare filters")
if is_reference:
    summary_table(rows)
    noisy_snr = float(rows[0]["SNR (dB)"])
    best_snr = next(float(row["SNR (dB)"]) for row in rows if row["Method"] == recommended)
    st.caption(f"Known clean reference · SNR gain with {recommended}: {best_snr - noisy_snr:+.2f} dB")
else:
    summary_table(rows)
    st.caption("Uploaded WAV: frequency-band proxies only; true SNR and MSE need a clean reference. High-frequency content may be useful audio.")

if any(name.startswith("Notch + FIR") for name in outputs):
    st.caption("Detected narrow 50/60 Hz tone · added a notch + FIR option. A musical bass note near that frequency may also trigger it; listen before exporting.")

st.success(f"**Suggested: {recommended}** · {reason}")
selected = st.selectbox("Listen to and export", list(outputs), index=list(outputs).index(recommended))
chosen = outputs[selected]
if np.max(np.abs(chosen)) > 1:
    st.warning("The selected output exceeds ±1. Playback/export is clipped to 16-bit WAV. Metrics and plots show the unclipped filter result.")

tab1, tab2 = st.tabs(["Signal analysis", "Listen & export"])
with tab1:
    st.markdown("**Waveform**")
    fig = waveform_figure(data.samples, chosen, data.sample_rate, profile_range)
    st.pyplot(fig, width="stretch")
    plt.close(fig)
    st.markdown("**Spectrogram · frequency over time**")
    fig = spectrogram_figure(data.samples, chosen, data.sample_rate)
    st.pyplot(fig, width="stretch")
    plt.close(fig)
    st.markdown("**FFT spectrum**")
    fig = spectrum_figure(data.samples, chosen, data.sample_rate, cutoff)
    st.pyplot(fig, width="stretch")
    plt.close(fig)
with tab2:
    a, b = st.columns(2)
    with a:
        st.markdown("**Original / noisy**")
        st.audio(wav_bytes(data.samples, data.sample_rate), format="audio/wav")
    with b:
        st.markdown(f"**{selected} output**")
        cleaned_wav = wav_bytes(chosen, data.sample_rate)
        st.audio(cleaned_wav, format="audio/wav")
        st.download_button("Download cleaned WAV", cleaned_wav, file_name=f"cleaned_{selected.lower().replace(' ', '_')}.wav", mime="audio/wav", type="primary")

with st.expander("Methods and viva notes"):
    st.markdown(
        """
        - **Moving average:** Replace each sample with the mean of neighboring samples. Simple smoothing, but its effective cutoff depends on the window length and is **not** the slider's IIR/FIR cutoff.
        - **Butterworth:** A fourth-order IIR low-pass with a smooth passband. It is run forward and backward so the offline result has no phase delay.
        - **FIR:** A Hamming-windowed finite impulse response low-pass. More taps produce a sharper transition but require more computation. It is also run forward and backward.
        - **Notch + FIR (when hum is detected):** A narrow IIR notch targets a strong 50 or 60 Hz tone, then FIR removes high-frequency hiss. The extra option appears only when the tone stands out clearly.
        - **Noise profile:** A short-time Fourier transform measures noise in a selected interval, then applies a smoothed magnitude mask. The selected interval must contain only noise; otherwise useful sound may be removed.
        - **Spectrogram:** Shows frequency changes over time, using the same color scale before and after.
        - **FFT:** Shows which frequencies are present. A low-pass alone suppresses hiss but generally leaves low-frequency hum.
        - **SNR and MSE:** Valid only when the generated clean signal is known. The app chooses the lowest-MSE filter there. Uploaded audio has no ground truth, so the app uses a stated high-band-removal versus low-band-preservation rule.
        """
    )
