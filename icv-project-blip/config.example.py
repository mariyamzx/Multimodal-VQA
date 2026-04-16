"""
Configuration file for VQA System
Contains all configurable parameters and API keys
"""

# ------------------- API KEYS -------------------
GEMINI_API_KEY = "your-gemini-api-key-here"  # Get from https://makersuite.google.com/app/apikey
GEN_AI_TIMEOUT = 10  # Seconds to wait for Gemini response before giving up
GEMINI_MODEL = "gemini-1.5-flash"  # Standard stable model (confirmed available)

# ------------------- CAMERA SETTINGS -------------------
# Camera source: Use 0 for default webcam, or IP camera URL
CAMERA_SOURCE = 0  # Change to your camera device ID
CAMERA_BUFFER_SIZE = 1  # Reduce buffer lag

# ------------------- MICROPHONE SETTINGS -------------------
MICROPHONE_DEVICE_ID = 0  # Run check_audio_ids.py to find your device ID
MICROPHONE_CHUNK_DURATION = 0.5  # Record in 0.5 second chunks for VAD
MICROPHONE_SILENCE_DURATION = 1.5  # Stop recording after 1.5 seconds of silence
MICROPHONE_MAX_DURATION = 10  # Maximum recording duration (safety limit)
MICROPHONE_FALLBACK_SAMPLE_RATE = 48000  # Hz

# ------------------- MODEL SETTINGS -------------------
WHISPER_MODEL = "tiny"  # Options: tiny, base, small, medium, large
BLIP_MODEL_ID = "Salesforce/blip-vqa-base"  # VQA Model
YOLO_MODEL = "yolov8n.pt"  # Options: yolov8n.pt, yolov8s.pt, yolov8m.pt
YOLO_CONFIDENCE_THRESHOLD = 0.4  # Detection confidence (0.0-1.0)

# ------------------- DETECTION SETTINGS -------------------
DETECTION_INTERVAL = 0.2  # Run YOLO every 0.2 seconds
IOU_THRESHOLD = 0.3  # Minimum IoU for object tracking
OBJECT_TRACKING_AGE_THRESHOLD = 2  # Minimum frames before showing object
OBJECT_TIMEOUT = 1.0  # Remove objects not seen for 1 second

# ------------------- TTS SETTINGS -------------------
TTS_RATE = 165  # Speech rate (words per minute)
TTS_VOLUME = 1.0  # Volume (0.0-1.0)

# ------------------- VOICE ACTIVITY DETECTION -------------------
VAD_THRESHOLD = 0.005  # Energy threshold for speech detection
VAD_SILENCE_THRESHOLD = 0.01  # Lower threshold for silence detection

# ------------------- FILTER SETTINGS -------------------
MIN_TEXT_LENGTH = 3  # Minimum transcribed text length
FILTER_WORDS = ["you", "thank you", "okay", "um", "uh", "hmm", "mm", "no no no"]

# ------------------- DISPLAY SETTINGS -------------------
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480
FPS_UPDATE_INTERVAL = 30  # Update FPS every N frames

# ------------------- LABEL MAPPING -------------------
# Map detected object names to custom names
LABEL_MAPPING = {
    "tv": "pc",
    # Add more mappings as needed, e.g.:
    # "laptop": "computer",
    # "cell phone": "mobile",
}
