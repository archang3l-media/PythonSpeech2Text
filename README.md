# GO AWAY VOICE MESSAGE

Since I'm in way too many meetings and get way too many voice messages on instant Messengers, I created a working Speech2Text Model in Python for processing voice messages.

### Features:
- Takes Audiofile as Input
- Extracts speech from the audio
- Uses OpenAI Whisper to transcribe the extracted speech to text

### ToDo:
- Wrap a nice API around everything
- Make it so that I can "share" voice messages with it and get a transcription back some time later
- Maybe even add bots for the messengers that take the voice messages, transcribe them, and notify me when transcriptions are available

#### Requirements:
- Except for the stuff in the requirements.txt
  - FFMPEG