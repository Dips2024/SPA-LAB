"""Run with: python -m unittest discover -s tests -v"""

import unittest

import numpy as np
from scipy.io import wavfile
from io import BytesIO

from ai import LABELS, predict_pattern, training_bank
from dsp import (
    AudioData,
    apply_filters,
    detect_hum,
    generate_test_signal,
    mse,
    proxy_scores,
    read_wav,
    recommend_with_reference,
    reference_scores,
    snr_db,
    spectral_gate,
    wav_bytes,
)


class DSPTests(unittest.TestCase):
    def test_noise_profile_improves_known_hiss_example(self):
        data = generate_test_signal(duration=2.0, noise_kind="High-frequency hiss")
        result = spectral_gate(data, (data.samples - data.clean_reference)[: data.sample_rate // 2])
        self.assertEqual(result.shape, data.samples.shape)
        self.assertTrue(np.all(np.isfinite(result)))
        self.assertLess(mse(data.clean_reference, result), mse(data.clean_reference, data.samples))

    def test_noise_profile_rejects_too_short_excerpt(self):
        data = generate_test_signal(duration=1.0)
        with self.assertRaisesRegex(ValueError, "0.1 seconds"):
            spectral_gate(data, data.samples[:100])

    def test_ai_pattern_on_unseen_synthetic_examples(self):
        bank = training_bank()
        for label in LABELS:
            sample = generate_test_signal(duration=1.5, noise_kind=label, seed=99)
            predicted, match = predict_pattern(sample, bank)
            self.assertEqual(predicted, label)
            self.assertGreater(match, 0.5)

    def test_synthetic_reference_and_recommendation(self):
        data = generate_test_signal(duration=1.0, noise_level=0.2)
        outputs = apply_filters(data, 1400, 7, 151)
        rows = reference_scores(data, outputs)
        name, _ = recommend_with_reference(rows)
        self.assertIn(name, outputs)
        self.assertTrue(all(np.isfinite(r["MSE"]) for r in rows))
        self.assertLess(min(r["MSE"] for r in rows[1:]), rows[0]["MSE"])
        self.assertGreater(max(r["SNR (dB)"] for r in rows[1:]), rows[0]["SNR (dB)"])

    def test_hum_adds_combined_filter_and_improves_reference(self):
        data = generate_test_signal(duration=3.0, noise_kind="50 Hz hum + hiss")
        self.assertEqual(detect_hum(data), 50)
        outputs = apply_filters(data, 1400, 7, 151)
        self.assertIn("Notch + FIR (50 Hz)", outputs)
        rows = reference_scores(data, outputs)
        self.assertEqual(recommend_with_reference(rows)[0], "Notch + FIR (50 Hz)")
        unknown = AudioData(data.samples, data.sample_rate, "uploaded.wav")
        self.assertEqual(proxy_scores(unknown, outputs, 1400)[1], "Notch + FIR (50 Hz)")

    def test_no_hum_keeps_three_core_filters(self):
        data = generate_test_signal(noise_kind="High-frequency hiss")
        self.assertIsNone(detect_hum(data))
        self.assertEqual(len(apply_filters(data, 1400)), 3)

    def test_detects_60_hz_hum(self):
        fs = 16000
        t = np.arange(3 * fs) / fs
        samples = (0.2 * np.sin(2 * np.pi * 60 * t) + 0.3 * np.sin(2 * np.pi * 440 * t))[:, None]
        data = AudioData(samples, fs, "hum60.wav")
        self.assertEqual(detect_hum(data), 60)
        self.assertIn("Notch + FIR (60 Hz)", apply_filters(data, 1400))

    def test_stereo_is_not_mixed(self):
        fs = 8000
        t = np.arange(fs) / fs
        left = 0.4 * np.sin(2 * np.pi * 300 * t)
        right = np.zeros_like(left)
        data = AudioData(np.column_stack([left, right]), fs, "stereo")
        for output in apply_filters(data, 1000).values():
            self.assertEqual(output.shape, data.samples.shape)
            self.assertLess(np.max(np.abs(output[:, 1])), 1e-10)

    def test_wav_round_trip_and_rejections(self):
        data = generate_test_signal(duration=1.0)
        decoded = read_wav(wav_bytes(data.samples, data.sample_rate))
        self.assertEqual(decoded.sample_rate, data.sample_rate)
        self.assertEqual(decoded.samples.shape, data.samples.shape)
        self.assertLess(np.max(np.abs(decoded.samples - data.samples)), 1e-4)
        with self.assertRaisesRegex(ValueError, "standard RIFF"):
            read_wav(b"not a wav")
        with self.assertRaisesRegex(ValueError, "Sample rate"):
            read_wav(wav_bytes(data.samples, 7000))
        with self.assertRaisesRegex(ValueError, "Audio must"):
            read_wav(wav_bytes(data.samples[:100], data.sample_rate))
        with self.assertRaisesRegex(ValueError, "silent"):
            read_wav(wav_bytes(np.zeros_like(data.samples), data.sample_rate))

    def test_float_wav_nonfinite_rejected(self):
        buf = BytesIO()
        wavfile.write(buf, 16000, np.array([np.nan] * 16000, dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            read_wav(buf.getvalue())

    def test_proxy_is_only_for_unknown_reference(self):
        data = generate_test_signal(duration=1.0, noise_level=0.2)
        unknown = AudioData(data.samples, data.sample_rate, "recording.wav")
        outputs = apply_filters(unknown, 1400)
        rows, name, reason = proxy_scores(unknown, outputs, 1400)
        self.assertEqual(len(rows), 3)
        self.assertIn(name, outputs)
        self.assertTrue(reason)
        self.assertNotIn("SNR", str(rows))
        self.assertTrue(all(r["High-band reduction (dB)"] <= 80 for r in rows))

    def test_metric_equivalence_and_no_improvement_warning(self):
        clean = np.ones((100, 1)) * 0.3
        noisy = clean + 0.1
        better = clean + 0.05
        self.assertAlmostEqual(snr_db(clean, better) - snr_db(clean, noisy), 10 * np.log10(mse(clean, noisy) / mse(clean, better)))
        rows = [
            {"Method": "Original noisy", "MSE": 0.01},
            {"Method": "Moving average", "MSE": 0.02},
            {"Method": "Butterworth", "MSE": 0.03},
        ]
        name, reason = recommend_with_reference(rows)
        self.assertEqual(name, "Moving average")
        self.assertIn("none improves", reason)


if __name__ == "__main__":
    unittest.main()
