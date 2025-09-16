#!/bin/env python

import os
# Use static FFmpeg to avoid library compatibility issues
static_ffmpeg_path = '/config/workspace/Benchmark/PythonSpeech2Text/libs/ffmpeg-7.0.2-amd64-static'
os.environ['PATH'] = f"{static_ffmpeg_path}:{os.environ.get('PATH', '')}"
os.environ['FFMPEG_BINARY'] = f"{static_ffmpeg_path}/ffmpeg"
os.environ['FFPROBE_BINARY'] = f"{static_ffmpeg_path}/ffprobe"

import time
from pathlib import Path
import torch
import whisper
import argparse
import librosa
import numpy as np
import os.path
import soundfile as sf
from scipy import signal


def preprocess_audio(input_path, output_path=None, apply_filters=True):
    """
    Preprocess audio to improve quality for speech recognition
    Parameters:
    - input_path: Path to input audio file
    - output_path: Path to save processed audio (optional)
    - apply_filters: Whether to apply noise reduction and filtering
    Returns:
    - Processed audio data and sample rate
    """
    print("Loading audio for preprocessing...")
    # Load the audio file
    audio, sr = librosa.load(input_path, sr=None)
    if apply_filters:
        print("Applying audio enhancement filters...")

        # High-pass filter to reduce low-frequency noise (including wind)
        # Cut off below 80Hz which helps with wind noise
        b, a = signal.butter(5, 80 / (sr / 2), 'highpass')
        audio = signal.filtfilt(b, a, audio)

        # Very simple Spectral noise gating for additional noise reduction
        frame_length = int(sr * 0.025)  # 25ms frames
        hop_length = int(sr * 0.010)  # 10ms hop

        # Compute spectrogram
        stft = librosa.stft(audio, n_fft=2048, hop_length=hop_length, win_length=frame_length)
        mag = np.abs(stft)

        # Estimate noise from relatively silent parts (adjust percentile as needed)
        noise_thresh = np.percentile(mag, 20, axis=1, keepdims=True)

        # Apply soft mask based on signal-to-noise ratio
        mask = (mag - noise_thresh) / mag
        mask = np.maximum(0, mask)
        mask = np.minimum(1, mask ** 2)  # Non-linear scaling for smoother sound

        # Apply mask and reconstruct signal
        stft_denoised = stft * mask
        audio = librosa.istft(stft_denoised, hop_length=hop_length, win_length=frame_length)

        # Normalize audio level
        audio = audio / np.max(np.abs(audio)) * 0.9

        if output_path:
            sf.write(output_path, audio, sr)
            print(f"Processed audio saved to {output_path}")

    return audio, sr


# Initialize ArgParser so we can give a filename via CLI
parser = argparse.ArgumentParser(description='Transcribe audio file using Whisper')
parser.add_argument('input_file', type=str, help='Path to the input audio file')
args = parser.parse_args()
input_path = Path(args.input_file)
# Perform transcription
try:
    # Check if a GPU Backend is available
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"  # Metal Performance Shaders (for Mac M1/M2)
        print("Using Apple Metal GPU acceleration")
    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
        device = "xpu"  # Intel XPU devices
        print("Using Intel XPU acceleration")
    elif hasattr(torch, 'hip') and torch.hip.is_available():
        device = "cuda"  # ROCm uses CUDA device type in PyTorch
        print(f"Using AMD ROCm GPU acceleration")
        print(f"GPU Name: {torch.hip.get_device_name(0)}")
    # least specific matching for generic CUDA
    elif torch.cuda.is_available():
        device = "cuda"
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    # If we are running on a really sad device
    else:
        device = "cpu"
        print("No GPU acceleration available, using CPU")
    print(f"Using device: {device}")
    # Preprocess audio
    total_runtime = time.time()
    start_time = time.time()
    print("Preprocessing audio...")
    temp_processed_path = os.path.join(os.path.dirname(input_path), "processed_audio.wav")
    audio, sr = preprocess_audio(input_path, temp_processed_path)
    processing_time = time.time() - start_time
    print(f"Audio preprocessed in {processing_time:.2f} seconds.")

    # Perform transcription with Whisper
    print("Starting transcription...")
    start_time = time.time()
    print(f"Loading Whisper model for {device}...")
    if device == "cpu":
        start_time = time.time()
        # Small modell to not overwhelm the CPU. Change at your own risk
        model = whisper.load_model("small", device=device)
        end_time = time.time()
        load_time = end_time - start_time
        # CPU processing with FP32, since FP16 usually is not available
        print(f"Whisper model optimized for {device} loaded successfully in {load_time:.2f} seconds.")
        result = model.transcribe(temp_processed_path, language="de", fp16=False)
    else:
        start_time = time.time()
        # I only have an RTX 3080, so no "large" model for me, unless I want to wait 150% of the audio runtime
        model = whisper.load_model("medium", device=device)
        end_time = time.time()
        load_time = end_time - start_time
        print(f"Whisper model optimized for {device} loaded successfully in {load_time:.2f} seconds.")
        result = model.transcribe(temp_processed_path, language="de", fp16=True)
    end_time = time.time()
    process_time = end_time - start_time
    text = result["text"]
    print(f"\nTranscription complete in {process_time:.2f} seconds:\n")
    runtime = time.time() - total_runtime
    print(text + "\n")
    print(f"\nTotal runtime: {runtime:.2f} seconds.")
except Exception as e:
    print(f"Error during transcription: {e}")
