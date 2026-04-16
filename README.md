# 🧠 Multimodal-VQA — Real-Time Visual Question Answering System

A real-time **Visual Question Answering (VQA)** system that combines computer vision, speech recognition, and AI language models to answer spoken questions about what the camera sees — all in real time.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green?logo=opencv)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)
![BLIP](https://img.shields.io/badge/BLIP-Salesforce-orange)
![Whisper](https://img.shields.io/badge/Whisper-OpenAI-black)
![Platform](https://img.shields.io/badge/Platform-Windows-blue?logo=windows)

---

## ✨ Features

- **🎯 Real-Time Object Detection** — YOLOv8-powered detection with smooth object tracking
- **🧠 Intelligent VLM Routing** — Automatically routes questions to **BLIP** (fast, local) or **Gemini** (advanced reasoning) based on complexity
- **🎤 Voice Interaction** — Ask questions by speaking; answers are spoken back via TTS
- **⚡ Async Architecture** — Multi-threaded, non-blocking design for 30–60 FPS performance
- **🔄 Smart Fallback** — If one model fails, automatically falls back to the other
- **📝 Context Injection** — Detected objects are injected as context to improve answer accuracy

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    USER SPEAKS A QUESTION                │
└────────────────────────┬─────────────────────────────────┘
                         ▼
              ┌─────────────────────┐
              │  Whisper STT (Tiny) │  ← Speech-to-Text
              └────────┬────────────┘
                       ▼
            ┌───────────────────────┐
            │  Question Classifier  │  ← Keyword + Heuristic
            └─────┬───────────┬─────┘
                  │           │
          SIMPLE  ▼           ▼  COMPLEX
     ┌────────────────┐  ┌────────────────┐
     │   BLIP VQA     │  │  Gemini VLM    │
     │  (Local, Fast) │  │ (Cloud, Smart) │
     └───────┬────────┘  └───────┬────────┘
             │    Fallback ↔     │
             └────────┬──────────┘
                      ▼
            ┌───────────────────┐
            │  Answer Formatter │  ← Natural language output
            └────────┬──────────┘
                     ▼
            ┌───────────────────┐
            │  Windows TTS      │  ← Speaks answer aloud
            └───────────────────┘
```

**Parallel Threads:**
| Thread | Purpose |
|---|---|
| Camera Thread | Continuous frame capture (no lag) |
| YOLO Detection Thread | Async object detection every 200ms |
| TTS Worker Thread | Non-blocking speech output |
| Main Display Loop | 30–60 FPS video with overlays |

---

## 📁 Project Structure

```
Multimodal-VQA/
├── main.py                  # Entry point — orchestrates all modules
├── config.example.py        # Example configuration (copy to config.py)
├── vision.py                # Camera capture + YOLOv8 detection + tracking
├── vqa_engine.py            # VQA engine with BLIP/Gemini routing
├── gemini_vlm.py            # Google Gemini Vision API integration
├── question_classifier.py   # Question complexity classifier
├── speech_to_text.py        # Whisper-based speech recognition with VAD
├── tts.py                   # Text-to-speech (Windows SAPI + fallback)
├── check_audio_ids.py       # Utility: list audio input devices
├── find_droidcam.py         # Utility: find DroidCam camera ID
├── list_models.py           # Utility: list available Gemini models
├── test_audio_conflict.py   # Utility: test audio device conflicts
├── run_app.bat              # Windows batch script to launch app
├── requirements.txt         # Python dependencies
├── MODULE_STRUCTURE.md      # Module documentation
└── REALTIME_ARCHITECTURE.md # Architecture documentation
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.8+**
- **Windows OS** (required for TTS via SAPI)
- **Webcam** or IP camera (e.g., DroidCam)
- **Microphone**
- **Google Gemini API Key** — [Get one here](https://makersuite.google.com/app/apikey)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/mariyamzx/Multimodal-VQA.git
   cd Multimodal-VQA/icv-project-blip
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up configuration**
   ```bash
   copy config.example.py config.py
   ```
   Then edit `config.py` and set:
   - `GEMINI_API_KEY` — Your Google Gemini API key
   - `CAMERA_SOURCE` — Camera device ID (`0` for default webcam)
   - `MICROPHONE_DEVICE_ID` — Microphone device ID

5. **Find your microphone device ID**
   ```bash
   python check_audio_ids.py
   ```

6. **Find your camera ID** (optional, for DroidCam users)
   ```bash
   python find_droidcam.py
   ```

---

## ▶️ Usage

### Run the system
```bash
python main.py
```

Or use the batch script:
```bash
run_app.bat
```

### How It Works

1. The system opens a live camera feed showing detected objects with bounding boxes
2. **Speak your question** into the microphone
3. The system will:
   - Transcribe your speech (Whisper)
   - Classify the question complexity
   - Route to BLIP (simple) or Gemini (complex)
   - Display and **speak** the answer
4. Press **`q`** to quit

### Example Questions

| Type | Example | Handled By |
|------|---------|------------|
| Simple | *"What do you see?"* | BLIP |
| Simple | *"How many people are there?"* | BLIP |
| Simple | *"What color is the laptop?"* | BLIP |
| Simple | *"Is there a phone on the desk?"* | BLIP |
| Complex | *"Why is the person doing that?"* | Gemini |
| Complex | *"What might happen next?"* | Gemini |
| Complex | *"Describe the mood of this scene"* | Gemini |
| Complex | *"Compare the objects on the left and right"* | Gemini |

---

## ⚙️ Configuration

All settings are in `config.py` (copy from `config.example.py`):

| Setting | Description | Default |
|---------|-------------|---------|
| `GEMINI_API_KEY` | Google Gemini API key | *(required)* |
| `CAMERA_SOURCE` | Camera device ID or URL | `0` |
| `MICROPHONE_DEVICE_ID` | Microphone device ID | `0` |
| `WHISPER_MODEL` | Whisper model size (`tiny`, `base`, `small`, `medium`, `large`) | `tiny` |
| `BLIP_MODEL_ID` | BLIP model identifier | `Salesforce/blip-vqa-base` |
| `YOLO_MODEL` | YOLO model file | `yolov8n.pt` |
| `YOLO_CONFIDENCE_THRESHOLD` | Detection confidence threshold | `0.4` |
| `DETECTION_INTERVAL` | YOLO detection frequency (seconds) | `0.2` |
| `TTS_RATE` | Speech rate (words per minute) | `165` |
| `VAD_THRESHOLD` | Voice activity detection energy threshold | `0.005` |

---

## 🧩 Module Overview

| Module | Description |
|--------|-------------|
| `main.py` | System orchestrator — coordinates all modules, manages threads |
| `vision.py` | Camera capture + YOLOv8 detection + IoU-based object tracking |
| `vqa_engine.py` | Intelligent VLM routing — BLIP for simple, Gemini for complex questions |
| `gemini_vlm.py` | Google Gemini Vision API with quota tracking and error handling |
| `question_classifier.py` | Keyword + heuristic question complexity classifier |
| `speech_to_text.py` | Whisper STT with Voice Activity Detection and silence-based recording |
| `tts.py` | Windows SAPI TTS with PowerShell fallback, async queue processing |

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| Display FPS | 30–60 FPS |
| Camera Latency | < 100ms |
| YOLO Detection | Every 200ms |
| TTS | Non-blocking (video continues during speech) |

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| Camera not detected | Change `CAMERA_SOURCE` in `config.py`. Try `0`, `1`, `2` |
| Microphone not working | Run `python check_audio_ids.py` and update `MICROPHONE_DEVICE_ID` |
| Low FPS | Increase `DETECTION_INTERVAL` or use smaller YOLO model |
| Gemini errors | Check API key, internet connection, and API quota |
| No audio output | Ensure Windows SAPI is available; check speaker output device |

---

## 📦 Dependencies

- **Computer Vision**: OpenCV, Ultralytics YOLOv8, NumPy, Pillow
- **Speech**: OpenAI Whisper, sounddevice, librosa
- **Text-to-Speech**: pyttsx3, pywin32 (Windows SAPI)
- **AI/ML**: Google Generative AI, Hugging Face Transformers, PyTorch

See [`requirements.txt`](requirements.txt) for the full list.

---

## 📄 License

This project is provided as-is for educational purposes.

## 🙏 Acknowledgments

- [**YOLOv8**](https://github.com/ultralytics/ultralytics) by Ultralytics — Object Detection
- [**OpenAI Whisper**](https://github.com/openai/whisper) — Speech Recognition
- [**Google Gemini**](https://ai.google.dev/) — Advanced Vision-Language Model
- [**Salesforce BLIP**](https://huggingface.co/Salesforce/blip-vqa-base) — Visual Question Answering
- [**pyttsx3**](https://github.com/nateshmbhat/pyttsx3) — Text-to-Speech
