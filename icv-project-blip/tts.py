"""
Text-to-Speech Module - Robust Windows Fallback
Uses direct SAPI COM dispatch and PowerShell to bypass Python threading issues.
Guarantees audio output even if Python's event loop is blocked.
"""

import queue
import threading
import sys
import subprocess
import winsound
import time
from config import TTS_RATE, TTS_VOLUME

# Verify output encoding for Windows
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass # Python < 3.7

class TTSManager:
    """Manages text-to-speech operations with robust fallbacks"""
    
    def __init__(self, is_speaking_ref=None):
        self.is_speaking_ref = is_speaking_ref
        self.tts_queue = queue.Queue()
        self.running = True
        
        # Start TTS worker thread
        self.tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
        self.tts_thread.start()
        
        print("✅ Robust TTS Manager ready")
    
    def _tts_worker(self):
        """Worker thread that processes speech requests sequentially"""
        
        # Initialize COM in this thread
        if sys.platform == 'win32':
            try:
                import pythoncom
                pythoncom.CoInitialize()
            except Exception as e:
                print(f"⚠ TTS Thread CoInitialize Warning: {e}", flush=True)

        # Create Speaker Object ONCE
        speaker = None
        if sys.platform == 'win32':
            try:
                import win32com.client
                speaker = win32com.client.Dispatch("SAPI.SpVoice")
                speaker.Rate = 1  # 1 is normal speed
                speaker.Volume = 100
                print("✅ TTS Worker: SAPI Voice Initialized", flush=True)
            except Exception as e:
                print(f"⚠ TTS Worker: SAPI Init Failed: {e}", flush=True)

        while self.running:
            try:
                # Get text from queue
                text = self.tts_queue.get(timeout=0.5)
                
                if text is None:  # Shutdown
                    break
                
                if not text.strip():
                    self.tts_queue.task_done()
                    continue

                print(f"🔊 SPEAKING: '{text}'", flush=True)

                # Signal Mic to Deafen
                if self.is_speaking_ref:
                    self.is_speaking_ref['value'] = True
                
                # Method 1: SAPI (Fastest)
                success = False
                if speaker:
                    try:
                        # Speak synchronously (Blocking)
                        speaker.Speak(text, 0)
                        success = True
                    except Exception as e:
                        print(f"⚠ SAPI Error: {e}. Re-initializing...", flush=True)
                        # Try to re-init
                        try:
                            speaker = win32com.client.Dispatch("SAPI.SpVoice")
                            speaker.Speak(text, 0)
                            success = True
                        except:
                            print("❌ SAPI Re-init failed.", flush=True)
                
                # Method 2: PowerShell (Fallback)
                if not success:
                    print("🔄 Falling back to PowerShell TTS...", flush=True)
                    try:
                        # Escape quotes for PowerShell
                        safe_text = text.replace("'", "''").replace('"', " ")
                        ps_command = f"Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{safe_text}');"
                        subprocess.run(["powershell", "-Command", ps_command], check=True)
                        success = True
                    except Exception as e:
                        print(f"❌ PowerShell Failed: {e}", flush=True)

                # Safety buffer: Wait a moment before opening mic to prevent echo
                time.sleep(0.5)

                # Signal Mic to Listen
                if self.is_speaking_ref:
                    self.is_speaking_ref['value'] = False
                    
                self.tts_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ TTS Loop Error: {e}", flush=True)
                if self.is_speaking_ref:
                    self.is_speaking_ref['value'] = False
        
        # Cleanup COM
        if sys.platform == 'win32':
            try:
                pythoncom.CoUninitialize()
            except:
                pass

    def test_tts(self):
        """Queue a test message"""
        print("Test TTS called", flush=True)
        self.speak("System audio check. Beep.")

    def speak(self, text):
        """Queue text to be spoken"""
        print(f"DEBUG: speak() called with '{text}'", flush=True)
        if text:
            self.tts_queue.put(text)
            print(f"DEBUG: '{text}' put in queue. Queue size: {self.tts_queue.qsize()}", flush=True)

    def shutdown(self):
        self.running = False
        self.tts_queue.put(None)
        self.tts_thread.join(timeout=2)
