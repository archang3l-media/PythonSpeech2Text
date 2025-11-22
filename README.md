# GO AWAY VOICE MESSAGE

Since I'm in way too many meetings and get way too many voice messages on instant Messengers, I created a working Speech2Text Model in Python for processing voice messages.

### Features:
- Takes Audiofile as Input
- Extracts speech from the audio
- Uses OpenAI Whisper to transcribe the extracted speech to text

### Installation:
1. Ensure Python 3.8+ is installed.
2. Install FFMPEG (required for audio processing):
   - On Windows: Download from [FFMPEG official site](https://ffmpeg.org/download.html) and add to PATH.
   - On macOS: Use `brew install ffmpeg`.
   - On Linux: Use your package manager, e.g., `sudo apt install ffmpeg`.
3. Install Python dependencies:
   - Run `pip install -r src/requirements.txt`.
   - For GPU support, uncomment the appropriate `--extra-index-url` in `src/requirements.txt` based on your hardware (NVIDIA, AMD, or Mac).

### Usage:
- Run the script from the command line: `python SpeechToText.py <path_to_audio_file>`.
- Supported audio formats: WAV, MP3, etc. (handled by librosa).
- The script preprocesses the audio, transcribes it to German text, and prints the result along with runtime info.
- Processed audio is temporarily saved as `processed_audio.wav` in the input file's directory.

### ToDo:
- Wrap a nice API around everything
- Make it so that I can "share" voice messages with it and get a transcription back some time later
- Maybe even add bots for the messengers that take the voice messages, transcribe them, and notify me when transcriptions are available

#### Requirements:
- Except for the stuff in the requirements.txt
  - FFMPEG
