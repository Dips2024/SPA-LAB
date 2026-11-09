"""Recreate the two tiny demonstration WAVs; no internet or microphone needed."""

from pathlib import Path

from dsp import generate_test_signal, wav_bytes


def main() -> None:
    out = Path(__file__).parent / "sample_data"
    out.mkdir(exist_ok=True)
    demo = generate_test_signal(duration=2.0, noise_level=0.22, seed=7)
    (out / "noisy_demo.wav").write_bytes(wav_bytes(demo.samples, demo.sample_rate))
    (out / "clean_reference_demo.wav").write_bytes(wav_bytes(demo.clean_reference, demo.sample_rate))
    print(f"Saved two WAV files in {out}")


if __name__ == "__main__":
    main()
