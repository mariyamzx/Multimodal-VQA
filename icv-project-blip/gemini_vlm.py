"""
Gemini VLM - For Complex Visual Question Answering
Uses Google's Gemini API for advanced reasoning tasks
"""

import google.generativeai as genai
from PIL import Image
import cv2
import numpy as np
from config import GEMINI_API_KEY, GEMINI_MODEL, GEN_AI_TIMEOUT

class GeminiVLM:
    """Manages complex VQA using Google Gemini Vision"""
    
    def __init__(self):
        print(f"🔄 Initializing Gemini VLM: {GEMINI_MODEL}...")
        try:
            # Configure Gemini API
            genai.configure(api_key=GEMINI_API_KEY)
            
            # Initialize the model
            self.model = genai.GenerativeModel(GEMINI_MODEL)
            
            # Track quota status
            self.quota_exhausted = False
            self.consecutive_failures = 0
            
            print("✅ Gemini VLM initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize Gemini VLM: {e}")
            self.quota_exhausted = True
            raise e
    
    def answer_question(self, question, detections, frame=None):
        """
        Generate answer using Gemini Vision API
        
        Args:
            question: User's question
            detections: YOLO detections for context
            frame: OpenCV frame (BGR format)
            
        Returns:
            str: Answer from Gemini or error message
        """
        if self.quota_exhausted:
            return None  # Signal to fallback to BLIP
            
        if frame is None:
            return "I can't see anything right now."
        
        try:
            # Convert BGR (OpenCV) to RGB (PIL)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            # Build context from YOLO detections
            context_str = ""
            if detections:
                unique_labels = list(set(d['label'] for d in detections))
                if unique_labels:
                    context_str = f"\nDetected objects in the scene: {', '.join(unique_labels)}"
            
            # Construct the prompt
            prompt = f"""You are a helpful vision assistant. Answer the following question about the image concisely and naturally.

Question: {question}{context_str}

Provide a clear, natural answer in 1-2 sentences. Be specific and helpful."""
            
            # Generate response with timeout
            response = self.model.generate_content(
                [prompt, pil_image],
                request_options={"timeout": GEN_AI_TIMEOUT}
            )
            
            # Reset failure counter on success
            self.consecutive_failures = 0
            
            # Extract and return the answer
            if response and response.text:
                return response.text.strip()
            else:
                print("⚠️ Gemini returned empty response")
                return None
                
        except Exception as e:
            error_msg = str(e).lower()
            
            # Check for quota/rate limit errors
            if "quota" in error_msg or "rate limit" in error_msg or "429" in error_msg:
                print(f"⚠️ Gemini quota exhausted or rate limited: {e}")
                self.consecutive_failures += 1
                
                # Mark as exhausted after 3 consecutive failures
                if self.consecutive_failures >= 3:
                    self.quota_exhausted = True
                    print("❌ Gemini marked as unavailable due to quota limits")
                
                return None  # Signal fallback to BLIP
            
            # Other errors
            print(f"❌ Gemini VQA Error: {e}")
            self.consecutive_failures += 1
            
            if self.consecutive_failures >= 5:
                self.quota_exhausted = True
                print("❌ Gemini marked as unavailable due to repeated failures")
            
            return None
    
    def is_available(self):
        """Check if Gemini is available for use"""
        return not self.quota_exhausted
    
    def reset_quota_status(self):
        """Reset quota status (useful for manual retry)"""
        self.quota_exhausted = False
        self.consecutive_failures = 0
        print("🔄 Gemini quota status reset")
