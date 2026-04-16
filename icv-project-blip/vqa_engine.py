"""
VQA Engine - Intelligent VLM Switching
Uses BLIP for simple questions, Gemini for complex questions
With Smart Context Injection & Natural Language Formatting
"""

import sys
import torch
import warnings
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
import cv2
import numpy as np
from config import BLIP_MODEL_ID

# Import VLM modules
from gemini_vlm import GeminiVLM
from question_classifier import QuestionComplexityClassifier

# Suppress warnings
warnings.filterwarnings("ignore")

class VQAEngine:
    """
    Intelligent VQA Engine with dual VLM support
    - BLIP-2: Fast, efficient for simple questions
    - Gemini: Advanced reasoning for complex questions
    """
    
    def __init__(self):
        print("🔄 Initializing Intelligent VQA Engine...")
        
        # Initialize BLIP (Primary VLM for simple questions)
        self._init_blip()
        
        # Initialize Gemini (Secondary VLM for complex questions)
        self._init_gemini()
        
        # Initialize question classifier
        self.classifier = QuestionComplexityClassifier()
        
        print("✅ VQA Engine ready with intelligent VLM switching")
    
    def _init_blip(self):
        """Initialize BLIP VQA model"""
        print(f"🔄 Loading BLIP VQA model: {BLIP_MODEL_ID}...")
        try:
            # Determine device
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"⚙ BLIP Device: {self.device.upper()}")
            
            # Load Processor and Model
            self.blip_processor = BlipProcessor.from_pretrained(BLIP_MODEL_ID)
            self.blip_model = BlipForQuestionAnswering.from_pretrained(BLIP_MODEL_ID).to(self.device)
            self.blip_model.eval()
            
            self.blip_available = True
            print("✅ BLIP Model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load BLIP model: {e}")
            self.blip_available = False
    
    def _init_gemini(self):
        """Initialize Gemini VLM"""
        try:
            self.gemini_vlm = GeminiVLM()
            self.gemini_available = True
        except Exception as e:
            print(f"⚠️ Gemini VLM not available: {e}")
            self.gemini_available = False
            self.gemini_vlm = None
    
    def answer_question(self, question, detections, frame=None):
        """
        Generate answer using intelligent VLM routing
        
        Flow:
        1. Classify question complexity
        2. Route to appropriate VLM (BLIP or Gemini)
        3. Fallback to alternative VLM if primary fails
        """
        if frame is None:
            return "I can't see anything right now."
        
        # Classify question complexity
        complexity = self.classifier.classify(question)
        print(f"🧠 Question Complexity: {complexity}")
        
        # Route based on complexity
        if complexity == "SIMPLE":
            # Try BLIP first for simple questions
            answer = self._answer_with_blip(question, detections, frame)
            
            # Fallback to Gemini if BLIP fails
            if answer is None and self.gemini_available:
                print("⚠️ BLIP failed, falling back to Gemini...")
                answer = self._answer_with_gemini(question, detections, frame)
            
        else:  # COMPLEX
            # Try Gemini first for complex questions
            answer = self._answer_with_gemini(question, detections, frame)
            
            # Fallback to BLIP if Gemini fails or quota exhausted
            if answer is None and self.blip_available:
                print("⚠️ Gemini unavailable, falling back to BLIP...")
                answer = self._answer_with_blip(question, detections, frame)
        
        # Final fallback
        if answer is None:
            return "I'm having trouble answering that question right now."
        
        return answer
    
    def _answer_with_gemini(self, question, detections, frame):
        """Answer using Gemini VLM"""
        if not self.gemini_available or self.gemini_vlm is None:
            return None
        
        if not self.gemini_vlm.is_available():
            print("⚠️ Gemini quota exhausted")
            return None
        
        print("🤖 Using Gemini VLM...")
        return self.gemini_vlm.answer_question(question, detections, frame)
    
    def _answer_with_blip(self, question, detections, frame):
        """Answer using BLIP VQA with context injection"""
        if not self.blip_available:
            return None
        
        print("🤖 Using BLIP VLM...")
        
        try:
            # Convert BGR (OpenCV) to RGB (PIL)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            raw_image = Image.fromarray(rgb_frame)
            
            # --- CONTEXT INJECTION ---
            context_str = ""
            if detections:
                unique_labels = list(set(d['label'] for d in detections))
                if unique_labels:
                    context_str = f"in the image there is {', '.join(unique_labels)}. "
                    print(f"🧠 Context Hint: {context_str}")
            
            # Formulate the prompt
            if context_str:
                final_prompt = f"Question: {question} (Hint: {context_str}) Answer:"
            else:
                final_prompt = f"Question: {question} Answer:"

            # Prepare inputs
            inputs = self.blip_processor(raw_image, final_prompt, return_tensors="pt").to(self.device)
            
            # Generate answer
            with torch.no_grad():
                out = self.blip_model.generate(**inputs, max_new_tokens=50)
                
            # Decode answer
            blip_answer = self.blip_processor.decode(out[0], skip_special_tokens=True)
            blip_answer = blip_answer.strip()
            
            # Format answer naturally
            final_answer = self._format_blip_answer(question, blip_answer, detections, context_str)
            return final_answer
            
        except Exception as e:
            print(f"❌ BLIP Inference Error: {e}")
            return None
    
    def _format_blip_answer(self, question, blip_answer, detections, context_str):
        """Format BLIP's raw answer into natural language"""
        final_answer = ""
        low_q = question.lower()
        
        # Check if question is generic
        is_generic = any(x in low_q for x in ["what do you see", "describe", "what is there", "what objects", "front of me", "around me"])
        
        # Acknowledge the scene for generic questions
        if detections and is_generic:
            unique_labels = list(set(d['label'] for d in detections))
            object_sentence = format_objects_list(unique_labels)
            final_answer += f"I see {object_sentence}. "
        
        # Format based on question type
        if "doing" in low_q and blip_answer.endswith("ing"):
            subject = "person" if "person" in context_str else "subject"
            final_answer += f"The {subject} is {blip_answer}."
        elif "color" in low_q or "colour" in low_q:
            final_answer += f"The color appears to be {blip_answer}."
        elif blip_answer.lower().startswith(("is ", "are ", "was ", "were ")):
            final_answer += f"It {blip_answer}"
        elif blip_answer.lower() in ['yes', 'no']:
            final_answer += f"{blip_answer.capitalize()}."
        elif len(blip_answer.split()) < 5:
            if blip_answer.isdigit():
                final_answer += f"The count is {blip_answer}."
            elif blip_answer.endswith("ing"):
                final_answer += f"The subject is {blip_answer}."
            else:
                final_answer += f"It appears to be {blip_answer}."
        else:
            final_answer += blip_answer
        
        return final_answer

def format_objects_list(items):
    """Helper to format list grammatically"""
    if not items: return ""
    cleaned_items = [f"a {item}" for item in items]
    if len(cleaned_items) == 1:
        return cleaned_items[0]
    elif len(cleaned_items) == 2:
        return f"{cleaned_items[0]} and {cleaned_items[1]}"
    else:
        return ", ".join(cleaned_items[:-1]) + f", and {cleaned_items[-1]}"
