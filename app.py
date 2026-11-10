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


st.set_page_config(page_title="Audio Noise Lab", page_icon="🎛️", layout="wide")
st.markdown(
    """
    <style>
      .stApp {background: #f4f7f8;}
      .block-container {max-width: 1200px; padding-top: 3.7rem; padding-bottom: 4rem;}
      h1, h2, h3 {color: #15313d; letter-spacing: -.025em;}
      h3 {padding-top: .3rem;}
      [data-testid="stSidebar"] {background: #ffffff; border-right: 1px solid #e5ecee;}
      [data-testid="stSidebar"] h2 {font-size: 1.15rem;}
      [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {gap: .7rem;}
      .stats-grid {display: grid; grid-template-columns: repeat(3,minmax(0,1fr)); gap: .8rem; margin-bottom: 1.5rem;}
      .stat-card {background: #fff; border: 1px solid #e1e9eb; border-radius: 15px; padding: .9rem 1.1rem; box-shadow: 0 4px 18px rgba(18,55,67,.04);}
      .stat-card span {display: block; color: #5a7179; font-size: .8rem; margin-bottom: .24rem;}
      .stat-card strong {display: block; color: #173743; font-size: 1.3rem; font-weight: 720; line-height: 1.2; overflow-wrap: anywhere;}
      [data-testid="stDataFrame"], [data-testid="stFileUploader"] {border-radius: 14px; overflow: hidden;}
      button[kind="primary"] {border-radius: 10px; font-weight: 650;}
      [data-baseweb="tab-list"] {gap: .35rem; border-bottom: 1px solid #dce6e8;}
      [data-baseweb="tab"] {font-weight: 650; border-radius: 10px 10px 0 0;}
      .hero {position: relative; display: flex; align-items: center; justify-content: space-between; min-height: 152px; overflow: hidden; background: linear-gradient(115deg,#153441 0%,#0b6569 100%); color: #fff; padding: 1.8rem 2rem; border-radius: 22px; margin-bottom: 1.25rem; box-shadow: 0 14px 34px rgba(12,80,83,.16);}
      .hero::after {content: ""; position: absolute; width: 270px; height: 270px; border: 1px solid rgba(255,255,255,.14); border-radius: 50%; right: -65px; top: -120px;}
      .hero-copy {position: relative; z-index: 1;}
      .hero-title {color: #fff; font-size: clamp(2rem,4vw,2.8rem); font-weight: 760; line-height: 1.12; letter-spacing: -.04em;}
      .hero-subtitle {margin-top: .5rem; color: #c9e9e6; font-size: .99rem;}
      .wave-mark {position: relative; z-index: 1; display: flex; align-items: center; gap: 6px; height: 64px; margin-right: 1.4rem; opacity: .83;}
      .wave-mark i {display: block; width: 6px; background: #8de1cf; border-radius: 20px;}
      .wave-mark i:nth-child(1),.wave-mark i:nth-child(7) {height: 16px;}
      .wave-mark i:nth-child(2),.wave-mark i:nth-child(6) {height: 34px;}
      .wave-mark i:nth-child(3),.wave-mark i:nth-child(5) {height: 52px;}
      .wave-mark i:nth-child(4) {height: 64px;}
      .source-card {display: flex; align-items: center; gap: .8rem; background: #fff; border: 1px solid #dce7e9; border-left: 4px solid #0d9488; border-radius: 15px; padding: .85rem 1rem; margin: .3rem 0 1rem; box-shadow: 0 4px 18px rgba(18,55,67,.04);}
      .source-icon {display: grid; place-items: center; flex: none; width: 38px; height: 38px; background: #e5f5f1; border-radius: 10px; color: #0b7772; font-size: 1.2rem;}
      .source-card span,.insight-label {display: block; font-size: .71rem; font-weight: 750; letter-spacing: .09em; text-transform: uppercase; color: #52717a;}
      .source-card strong {display: block; color: #163344; font-size: 1.09rem; overflow-wrap: anywhere;}
      .insight-card {background: #fff; border: 1px solid #dce7e9; border-radius: 16px; padding: 1.15rem 1.25rem; min-height: 148px; box-shadow: 0 4px 18px rgba(18,55,67,.04);}
      .insight-card.recommended {background: #edf9f5; border-color: #bce4d8;}
      .insight-title {color: #153743; font-size: 1.2rem; font-weight: 750; line-height: 1.3; margin: .45rem 0 .35rem; overflow-wrap: anywhere;}
      .insight-card p {color: #4f6971; font-size: .89rem; line-height: 1.5; margin: 0;}
      .section-lead {color: #5b737a; font-size: .93rem; margin: -.35rem 0 .75rem;}
      @media (max-width: 700px) {.block-container {padding-top: 4.5rem; padding-left: 1rem; padding-right: 1rem;} .hero {padding: 1.45rem; min-height: 128px;} .hero-subtitle {font-size: .88rem;} .wave-mark {display: none;} .insight-card {min-height: 0;} .stat-card {padding: .75rem;} .stat-card strong {font-size: 1.05rem;}}
      @media (max-width: 390px) {.stats-grid {grid-template-columns: repeat(2,minmax(0,1fr));} .stat-card:last-child {grid-column: span 2;}}
    </style>
    <div class="hero"><div class="hero-copy"><div class="hero-title">Audio Noise Lab</div><div class="hero-subtitle">Clean, compare, and export audio.</div></div><div class="wave-mark" aria-hidden="true"><i></i><i></i><i></i><i></i><i></i><i></i><i></i></div></div>
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
    number_columns = {
        key: st.column_config.NumberColumn(key, format="%.7f" if key == "MSE" else "%.2f")
        for key in rows[0]
        if key != "Method"
    }
    st.dataframe(rows, column_config=number_columns, width="stretch", hide_index=True)


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

st.markdown(
    f'<div class="source-card"><div class="source-icon" aria-hidden="true">♫</div><div><span>Current recording</span><strong>{escape(data.name)}</strong></div></div>',
    unsafe_allow_html=True,
)
channels = "Mono" if data.samples.shape[1] == 1 else "Stereo"
st.markdown(
    f'<div class="stats-grid"><div class="stat-card"><span>Duration</span><strong>{data.duration:.2f} s</strong></div>'
    f'<div class="stat-card"><span>Sample rate</span><strong>{data.sample_rate:,} Hz</strong></div>'
    f'<div class="stat-card"><span>Channels</span><strong>{channels}</strong></div></div>',
    unsafe_allow_html=True,
)

pattern, match = predict_pattern(data, get_ai_bank())
st.subheader("Processing insights")
ai_col, suggestion_col = st.columns(2)
with ai_col:
    st.markdown(
        f'<div class="insight-card"><span class="insight-label">Detected noise pattern</span><div class="insight-title">{escape(pattern)}</div><p>{match:.0%} template vote share · Offline analysis</p></div>',
        unsafe_allow_html=True,
    )
with suggestion_col:
    st.markdown(
        f'<div class="insight-card recommended"><span class="insight-label">Suggested filter</span><div class="insight-title">{escape(recommended)}</div><p>{escape(reason)}</p></div>',
        unsafe_allow_html=True,
    )
with st.expander("About the AI result"):
    st.caption("An offline 5-nearest-neighbors model compares six spectral features with 48 generated, labeled examples. The match is a vote share, not a calibrated probability. Music or speech may resemble a noise pattern; the quality table determines the filter suggestion.")

st.subheader("Explore an output")
selected = st.selectbox("Filter to preview", list(outputs), index=list(outputs).index(recommended))
chosen = outputs[selected]
if np.max(np.abs(chosen)) > 1:
    st.warning("The selected output exceeds ±1. Playback/export is clipped to 16-bit WAV. Metrics and plots show the unclipped filter result.")

compare_tab, analysis_tab, listen_tab = st.tabs(["Compare filters", "Signal analysis", "Listen & download"])
with compare_tab:
    st.markdown('<p class="section-lead">Compare each method on the same recording.</p>', unsafe_allow_html=True)
    summary_table(rows)
    if is_reference:
        noisy_snr = float(rows[0]["SNR (dB)"])
        best_snr = next(float(row["SNR (dB)"]) for row in rows if row["Method"] == recommended)
        st.caption(f"Known clean reference · SNR gain with {recommended}: {best_snr - noisy_snr:+.2f} dB")
    else:
        st.caption("Uploaded WAV: frequency-band proxies only; true SNR and MSE need a clean reference. High-frequency content may be useful audio.")
    if any(name.startswith("Notch + FIR") for name in outputs):
        st.caption("A narrow 50/60 Hz tone was detected, so a notch + FIR option was added. Listen before exporting; a bass note can also trigger it.")
with analysis_tab:
    waveform_tab, spectrogram_tab, spectrum_tab = st.tabs(["Waveform", "Spectrogram", "FFT spectrum"])
    with waveform_tab:
        fig = waveform_figure(data.samples, chosen, data.sample_rate, profile_range)
        st.pyplot(fig, width="stretch")
        plt.close(fig)
    with spectrogram_tab:
        fig = spectrogram_figure(data.samples, chosen, data.sample_rate)
        st.pyplot(fig, width="stretch")
        plt.close(fig)
    with spectrum_tab:
        fig = spectrum_figure(data.samples, chosen, data.sample_rate, cutoff)
        st.pyplot(fig, width="stretch")
        plt.close(fig)
with listen_tab:
    st.markdown('<p class="section-lead">Listen before saving your cleaned WAV.</p>', unsafe_allow_html=True)
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
