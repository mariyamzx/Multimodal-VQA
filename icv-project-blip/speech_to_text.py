"""
Speech-to-Text Module with End-of-Speech Detection
Handles microphone input and Whisper speech recognition
Records until user finishes speaking (silence detected)
"""

import time
import threading
import whisper
import sounddevice as sd
import numpy as np
from collections import deque
from config import (
    MICROPHONE_DEVICE_ID,
    MICROPHONE_CHUNK_DURATION,
    MICROPHONE_SILENCE_DURATION,
    MICROPHONE_MAX_DURATION,
    WHISPER_MODEL,
    VAD_THRESHOLD,
    VAD_SILENCE_THRESHOLD,
    MIN_TEXT_LENGTH,
    FILTER_WORDS
)

# Hardcoded for Whisper
SAMPLE_RATE = 16000 

def get_microphone_device(device_id):
    """Get device info"""
    print("\n🎧 AVAILABLE AUDIO DEVICES:")
    print(sd.query_devices())
    print("\n")
    try:
        if device_id is None:
            device_info = sd.query_devices(kind='input')
        else:
            device_info = sd.query_devices(device_id, 'input')
        
        print(f"✅ Using Device: {device_info['name']}")
        return device_info
    except Exception as e:
        print(f"⚠ Audio Device Error: {e}")
        return None

def has_speech(audio_data, threshold=VAD_THRESHOLD):
    """Simple VAD to detect if audio contains speech"""
    energy = np.sqrt(np.mean(audio_data**2))
    return energy > threshold

class SpeechToText:
    """Manages speech recognition using Whisper with end-of-speech detection"""
    
    def __init__(self, question_callback, processing_lock, question_lock, is_processing_ref, is_speaking_ref):
        self.question_callback = question_callback
        self.processing_lock = processing_lock
        self.question_lock = question_lock
        self.is_processing_ref = is_processing_ref
        self.is_speaking_ref = is_speaking_ref
        
        # Setup microphone
        get_microphone_device(MICROPHONE_DEVICE_ID)
        sd.default.device = (MICROPHONE_DEVICE_ID, None)
        sd.default.samplerate = SAMPLE_RATE
        sd.default.channels = 1
        
        # Load Whisper model
        print("Loading Whisper model...")
        self.whisper_model = whisper.load_model(WHISPER_MODEL)
        print("✅ Whisper loaded")
        
        self.listener_thread = None
        self.running = True # Control flag
    
    def start_listening(self, speak_callback):
        """Start the microphone listener thread"""
        speak_callback("I'm listening")
        
        self.listener_thread = threading.Thread(
            target=self._mic_listener,
            daemon=True
        )
        self.listener_thread.start()
    
    def _record_until_silence(self):
        """
        Record audio until end of speech is detected (silence after speech)
        Returns: numpy array of audio data, or None if no speech detected
        """
        chunk_samples = int(MICROPHONE_CHUNK_DURATION * SAMPLE_RATE)
        audio_chunks = []
        speech_detected = False
        silence_start_time = None
        recording_start_time = time.time()
        
        # print("🎤 Listening... (speak now)")
        
        while self.running:
            # 🛑 CRITICAL: Abort recording if system starts speaking
            if self.is_speaking_ref and self.is_speaking_ref['value']:
                return None

            # Check max duration
            if time.time() - recording_start_time > MICROPHONE_MAX_DURATION:
                break
            
            # Record a chunk
            try:
                chunk = sd.rec(
                    chunk_samples,
                    samplerate=SAMPLE_RATE,
                    channels=1,
                    dtype="float32"
                )
                sd.wait()
                chunk = chunk.flatten()
                
                # Check if there's speech in this chunk
                if has_speech(chunk):
                    if not speech_detected:
                        speech_detected = True
                    
                    # Reset silence timer
                    silence_start_time = None
                    audio_chunks.append(chunk)
                else:
                    # No speech in this chunk
                    if speech_detected:
                        # We've detected speech before, check for silence
                        if silence_start_time is None:
                            silence_start_time = time.time()
                        
                        # Check if we've had enough silence
                        if time.time() - silence_start_time >= MICROPHONE_SILENCE_DURATION:
                            break
                        
                        # Still add chunk (might be pause in speech)
                        audio_chunks.append(chunk)
                    else:
                        # No speech detected yet, continue listening
                        continue
                        
            except Exception as e:
                print(f"❌ Recording error: {e}")
                return None
        
        if not speech_detected or len(audio_chunks) == 0:
            return None
        
        return np.concatenate(audio_chunks)
    
    def _mic_listener(self):
        """Continuous microphone listener loop"""
        while self.running:
            try:
                # 🛑 CRITICAL: Don't listen while the system itself is speaking
                if self.is_speaking_ref['value']:
                    time.sleep(0.1)
                    continue

                # Record until silence is detected
                audio = self._record_until_silence()
                
                if audio is None:
                    continue
                
                # Transcribe with Whisper (No resampling needed! audio is already 16k)
                result = self.whisper_model.transcribe(
                    audio,
                    fp16=False,
                    language="en"
                )
                text = result["text"].strip()
                
                # Filter out noise and very short utterances
                if len(text) < MIN_TEXT_LENGTH:
                    continue
                
                # Filter common filler words
                if text.lower() in FILTER_WORDS:
                    continue
                
                print(f"🗣 USER SAID: {text}")
                
                # Store question if not currently processing
                with self.processing_lock:
                    if not self.is_processing_ref['value']:
                        with self.question_lock:
                            self.question_callback(text)
                        self.is_processing_ref['value'] = True
                    else:
                        print("⏳ Still processing previous question")
                        
            except Exception as e:
                print(f"❌ Mic Error: {e}")
                time.sleep(1)
    
    def shutdown(self):
        self.running = False
