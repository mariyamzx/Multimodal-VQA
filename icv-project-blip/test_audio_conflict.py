
import time
import threading
import sounddevice as sd
from tts import TTSManager
from config import MICROPHONE_DEVICE_ID

print("=== Audio Conflict Test ===", flush=True)

# 1. Simulate SpeechToText initialization
print(f"Initializing SoundDevice with Input Device {MICROPHONE_DEVICE_ID}...", flush=True)
try:
    # Mimic speech_to_text.py init
    sd.default.device = (MICROPHONE_DEVICE_ID, None)
    print("SoundDevice initialized.", flush=True)
except Exception as e:
    print(f"Error initializing SoundDevice: {e}", flush=True)

# 2. Simulate TTS Initialization
print("Initializing TTSManager...", flush=True)
is_speaking = {'value': False}
tts = TTSManager(is_speaking_ref=is_speaking)
tts.test_tts()

print("TTS thread started. Queuing messages...", flush=True)
tts.speak("This is a test with SoundDevice initialized.")

# 3. Simulate Mic usage (briefly)
print("Simulating Mic recording for 1 second...", flush=True)
try:
    recording = sd.rec(int(44100 * 1), samplerate=44100, channels=1)
    sd.wait()
    print("Recording finished.", flush=True)
except Exception as e:
    print(f"Recording check failed: {e}", flush=True)

tts.speak("Audio recording finished. This should be audible.")

# Keep alive
for i in range(5):
    print(f"Main thread waiting {i}...", flush=True)
    time.sleep(1)

tts.shutdown()
print("Done.", flush=True)
