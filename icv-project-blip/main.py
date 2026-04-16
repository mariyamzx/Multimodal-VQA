"""
Main Entry Point for VQA System - Real-time Performance Optimized
Orchestrates all modules: Vision, Speech-to-Text, VQA, and TTS
Fast display loop with async processing
"""

import cv2
import threading
import time
from tts import TTSManager
from speech_to_text import SpeechToText
from vision import VisionSystem
from vqa_engine import VQAEngine
 

class VQASystem:
    """Main VQA System orchestrator with real-time performance"""
    
    def __init__(self):
        # Initialize all modules
        print("\n🚀 SYSTEM STARTING...\n")
        
        self.is_processing = {'value': False}  # Use dict for shared reference
        self.is_speaking = {'value': False}    # NEW: Flag to track if TTS is active
        self.last_user_question = None

        # Initialize TTS (now with async worker thread)
        self.tts = TTSManager(is_speaking_ref=self.is_speaking)
        self.tts.test_tts()
        
        # Initialize Vision System (now with async camera and detection threads)
        self.vision = VisionSystem()
        
        # Initialize VQA Engine
        self.vqa = VQAEngine()
        
        # Thread locks for synchronization
        self.processing_lock = threading.RLock()
        self.question_lock = threading.RLock()
        
        # Initialize Speech-to-Text
        self.speech_to_text = SpeechToText(
            question_callback=self._on_question_detected,
            processing_lock=self.processing_lock,
            question_lock=self.question_lock,
            is_processing_ref=self.is_processing,
            is_speaking_ref=self.is_speaking  # Pass the flag
        )
        
        # Start listening for speech
        self.speech_to_text.start_listening(self.tts.speak)
    
    def _on_question_detected(self, question):
        """Callback when a question is detected from speech"""
        with self.question_lock:
            self.last_user_question = question
            # Update UI Status
            self.vision.set_status("THINKING...", (0, 255, 255))  # Cyan

    def _process_vqa(self, question, frame):
        """Process VQA in background thread"""
        try:
            # Update UI Status
            self.vision.set_status("ANALYZING...", (0, 165, 255)) # Orange
            print(f"💭 User: {question}")
            
            # Get latest YOLO detections (flattened list)
            detections = self.vision.get_latest_yolo_result()
            
            # Pass frame for color analysis
            answer = self.vqa.answer_question(question, detections, frame)
            
            if answer is None:
                # Question ignored (noise)
                print("🔇 Ignored non-question")
                self.vision.set_status("LISTENING", (0, 255, 0))
                return

            print(f"🤖 System: {answer}\n")
            
            # Update UI
            self.vision.set_answer(answer)
            self.vision.set_status("SPEAKING", (255, 100, 100)) # Blue-ish
            
            # Queue answer for speaking
            self.tts.speak(answer)
            
        except Exception as e:
            print(f"❌ VQA Error: {e}")
            self.vision.set_status("ERROR", (0, 0, 255))
            self.vision.set_answer("System Error")
            self.tts.speak("Sorry, I encountered an internal error.")
            
        finally:
            # CRITICAL: Always reset processing flag
            with self.processing_lock:
                self.is_processing['value'] = False
                
                # Reset Status
                self.vision.set_status("LISTENING", (0, 255, 0))
                
                # AUDITORY CUE: Beep to signal "Ready to Listen"
                try:
                    import winsound
                    winsound.Beep(800, 100) # High-pitched beep
                except:
                    pass
    
    def run(self):
        """Main system loop - Fast display loop, all heavy processing is async"""
        try:
            # Fast display loop - only reads latest frame and displays
            while True:
                # Get latest frame (non-blocking, from camera thread)
                frame = self.vision.get_latest_frame()
                
                if frame is None:
                    time.sleep(0.01)  # Small sleep if no frame yet
                    continue
                
                # Draw detections (reads from detection thread results)
                self.vision.draw_detections(frame)
                
                # Update FPS
                self.vision.update_fps()
                
                # Draw info overlay
                self.vision.draw_info(frame)
                
                # Check for user question (non-blocking)
                with self.question_lock:
                    if self.last_user_question:
                        question = self.last_user_question
                        self.last_user_question = None
                        
                        # Get current frame for VLM processing
                        frame_for_vqa = self.vision.get_frame_for_vqa()
                        
                        # Process in background thread (non-blocking)
                        threading.Thread(
                            target=self._process_vqa,
                            args=(question, frame_for_vqa),
                            daemon=True
                        ).start()
                
                # Display frame (fast operation)
                cv2.imshow("VQA System - Press 'q' to quit", frame)
                
                # Exit on 'q' (non-blocking check)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # Small sleep to prevent CPU spinning (display loop runs at ~30-60 FPS)
                time.sleep(0.016)  # ~60 FPS max display rate
        
        except KeyboardInterrupt:
            print("\n⚠ Interrupted by user")
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Cleanup and shutdown system"""
        print("\n🛑 Shutting down...")
        self.vision.release()
        self.tts.shutdown()
        print("✅ System closed.")


if __name__ == "__main__":
    # Create and run VQA system
    system = VQASystem()
    system.run()
