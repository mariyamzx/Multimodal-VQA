# Modular Project Structure

## Overview
The project has been refactored into a clean, modular architecture with separate modules for each component.

## Project Structure

```
ICV-PROJ/
├── main.py              # Entry point - orchestrates all modules
├── config.py            # Configuration file (API keys, settings)
├── tts.py              # Text-to-Speech module
├── speech_to_text.py   # Speech recognition module (Whisper)
├── vision.py           # Camera and YOLO detection module
├── vqa_engine.py       # Gemini VLM VQA module
├── vqa.py              # Old monolithic file (can be removed)
├── requirements.txt    # Dependencies
└── README.md           # Documentation
```

## Module Descriptions

### 1. `config.py`
- **Purpose**: Centralized configuration
- **Contains**: 
  - API keys (Gemini)
  - Camera settings
  - Microphone settings
  - Model configurations
  - Detection parameters
  - TTS settings

### 2. `tts.py`
- **Purpose**: Text-to-Speech functionality
- **Class**: `TTSManager`
- **Methods**:
  - `test_tts()` - Test TTS functionality
  - `speak(text)` - Queue text for speech
  - `process_queue()` - Process TTS queue on main thread

### 3. `speech_to_text.py`
- **Purpose**: Speech recognition using Whisper
- **Class**: `SpeechToText`
- **Methods**:
  - `start_listening(speak_callback)` - Start microphone listener
  - `set_processing(value)` - Set processing state
- **Features**:
  - Voice Activity Detection (VAD)
  - Automatic microphone sample rate detection
  - Noise filtering

### 4. `vision.py`
- **Purpose**: Camera and object detection
- **Class**: `VisionSystem`
- **Methods**:
  - `get_frame()` - Get current camera frame
  - `detect_objects(frame)` - Run YOLO detection
  - `draw_detections(frame)` - Draw bounding boxes
  - `get_frame_for_vqa()` - Get frame for VQA (thread-safe)
  - `get_detected_objects()` - Get list of detected objects
  - `release()` - Cleanup resources
- **Features**:
  - Object tracking with IoU matching
  - Smooth bounding box updates
  - FPS calculation

### 5. `vqa_engine.py`
- **Purpose**: Visual Question Answering with Gemini VLM
- **Class**: `VQAEngine`
- **Methods**:
  - `answer_question(question, frame)` - Generate answer using VLM
- **Features**:
  - Image + text input to Gemini
  - BGR to RGB conversion
  - Error handling

### 6. `main.py`
- **Purpose**: Entry point and system orchestrator
- **Class**: `VQASystem`
- **Methods**:
  - `__init__()` - Initialize all modules
  - `run()` - Main system loop
  - `shutdown()` - Cleanup
- **Features**:
  - Coordinates all modules
  - Thread management
  - Main event loop

## How to Run

```bash
# Activate virtual environment
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac

# Run the system
python main.py
```

## Module Dependencies

```
main.py
├── config.py (all modules import this)
├── tts.py
├── speech_to_text.py
│   └── config.py
├── vision.py
│   └── config.py
└── vqa_engine.py
    └── config.py
```

## Benefits of Modular Structure

1. **Separation of Concerns**: Each module has a single responsibility
2. **Maintainability**: Easy to update individual components
3. **Testability**: Each module can be tested independently
4. **Reusability**: Modules can be reused in other projects
5. **Readability**: Clear structure, easy to understand
6. **Configuration**: Centralized settings in config.py

## Migration Notes

- The old `vqa.py` file is preserved but not used
- All functionality has been moved to modular components
- Configuration is now centralized in `config.py`
- The system works exactly the same, just better organized

