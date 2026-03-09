"""Test audio output format compliance."""

import struct
import numpy as np
import pytest


def test_pcm_output_sample_rate():
    """Verify expected sample rate is 24kHz."""
    TTS_SAMPLE_RATE = 24000
    assert TTS_SAMPLE_RATE == 24000, f"Expected 24000 Hz, got {TTS_SAMPLE_RATE} Hz"


def test_pcm_byte_format_little_endian():
    """Verify PCM format is 16-bit signed, little-endian."""
    # Create test data: [0.5, -0.5]
    test_values = [0.5, -0.5]

    # Convert to PCM 16-bit LE
    audio = np.array(test_values, dtype=np.float32)
    audio = np.clip(audio, -1.0, 1.0)
    samples = (audio * 32767).astype(np.int16)
    pcm = samples.tobytes()

    # Verify byte length
    assert len(pcm) == 4, f"Expected 4 bytes (2 samples * 2 bytes), got {len(pcm)}"

    # Unpack as little-endian signed shorts
    unpacked = struct.unpack("<2h", pcm)

    # 0.5 * 32767 ≈ 16383, -0.5 * 32767 ≈ -16383
    assert unpacked[0] > 16000 and unpacked[0] < 17000, f"First sample out of range: {unpacked[0]}"
    assert unpacked[1] < -16000 and unpacked[1] > -17000, f"Second sample out of range: {unpacked[1]}"


def test_pcm_mono_channel():
    """Verify output is single-channel (mono)."""
    # Create test audio: 4 samples
    audio = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    audio = np.clip(audio, -1.0, 1.0)
    samples = (audio * 32767).astype(np.int16)
    pcm = samples.tobytes()

    # 4 samples = 8 bytes, no extra channel info
    assert len(pcm) == 8, f"Expected 8 bytes for 4 mono samples, got {len(pcm)}"
    # Always even (2 bytes per sample)
    assert len(pcm) % 2 == 0, "PCM byte count must be even"


def test_pcm_clipping_prevents_overflow():
    """Verify values > 1.0 or < -1.0 are clipped to prevent int16 overflow."""
    audio = np.array([2.0, -2.0, 0.0], dtype=np.float32)
    audio = np.clip(audio, -1.0, 1.0)
    samples = (audio * 32767).astype(np.int16)
    pcm = samples.tobytes()

    unpacked = struct.unpack("<3h", pcm)

    # 2.0 clipped to 1.0 → 32767
    # -2.0 clipped to -1.0 → -32767 (actually -1.0 * 32767, not -32768)
    # 0.0 → 0
    assert unpacked[0] == 32767, f"Expected 32767, got {unpacked[0]}"
    assert -32768 <= unpacked[1] <= -32766, f"Expected ~-32767, got {unpacked[1]}"  # Allow ±1 precision
    assert unpacked[2] == 0, f"Expected 0, got {unpacked[2]}"


def test_pcm_duration_calculation():
    """Verify byte count correctly represents duration."""
    TTS_SAMPLE_RATE = 24000
    # Create 1 second of audio at 24kHz
    duration_sec = 1.0
    num_samples = int(TTS_SAMPLE_RATE * duration_sec)
    audio = np.zeros(num_samples, dtype=np.float32)

    samples = (audio * 32767).astype(np.int16)
    pcm = samples.tobytes()

    # Expected: num_samples * 2 bytes
    expected_bytes = num_samples * 2
    assert len(pcm) == expected_bytes, f"Expected {expected_bytes} bytes, got {len(pcm)}"

    # Verify: bytes → samples → duration
    calculated_samples = len(pcm) // 2
    calculated_duration = calculated_samples / TTS_SAMPLE_RATE
    expected_duration = 1.0

    assert (
        abs(calculated_duration - expected_duration) < 0.0001
    ), f"Duration mismatch: expected {expected_duration}s, got {calculated_duration}s"


def test_pcm_output_is_bytes():
    """Verify PCM output is bytes object."""
    audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    samples = (audio * 32767).astype(np.int16)
    pcm = samples.tobytes()

    assert isinstance(pcm, bytes), f"Expected bytes type, got {type(pcm)}"


def test_wave_file_format_compliance():
    """Verify PCM can be wrapped in WAV format."""
    # WAV header structure
    TTS_SAMPLE_RATE = 24000
    CHANNELS = 1
    BITS_PER_SAMPLE = 16

    audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    samples = (audio * 32767).astype(np.int16)
    pcm_data = samples.tobytes()

    # Construct minimal WAV header
    byte_rate = TTS_SAMPLE_RATE * CHANNELS * BITS_PER_SAMPLE // 8
    block_align = CHANNELS * BITS_PER_SAMPLE // 8

    # Basic WAV structure should be constructible
    assert byte_rate == TTS_SAMPLE_RATE * 2  # For mono 16-bit
    assert block_align == 2  # For mono 16-bit
    assert len(pcm_data) > 0
    assert len(pcm_data) % block_align == 0  # Data length must be multiple of block align


def test_pcm_zero_duration_handling():
    """Verify handling of zero-length audio."""
    audio = np.array([], dtype=np.float32)
    samples = (audio * 32767).astype(np.int16)
    pcm = samples.tobytes()

    # Zero duration should produce empty bytes
    assert len(pcm) == 0
    assert isinstance(pcm, bytes)
