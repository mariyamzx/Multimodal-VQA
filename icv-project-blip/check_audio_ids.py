import sounddevice as sd
import sys

sys.stdout.reconfigure(encoding='utf-8')

print("\n🎧 AVAILABLE AUDIO DEVICES:")
try:
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            print(f"ID {i}: {dev['name']}")
except Exception as e:
    print(f"Error listing devices: {e}")
print("\n")
