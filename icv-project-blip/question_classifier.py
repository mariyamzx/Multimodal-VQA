"""
Question Complexity Classifier
Determines whether a question is simple (BLIP can handle) or complex (needs Gemini)
"""

class QuestionComplexityClassifier:
    """Classifies questions as SIMPLE or COMPLEX for VLM routing"""
    
    # Keywords that indicate simple questions BLIP can handle
    SIMPLE_KEYWORDS = [
        # Object identification
        "what is this", "what is that", "what are these", "what are those",
        "what do you see", "what objects", "what items",
        
        # Counting
        "how many", "count",
        
        # Colors
        "what color", "what colour",
        
        # Basic attributes
        "is this", "is that", "is there", "are there",
        
        # Simple actions
        "what is the person doing", "what is he doing", "what is she doing",
        
        # Location (simple)
        "where is the", "where are the",
    ]
    
    # Keywords that indicate complex questions requiring Gemini
    COMPLEX_KEYWORDS = [
        # Reasoning & Analysis
        "why", "how come", "explain", "reason",
        
        # Comparison
        "compare", "difference between", "similar", "different",
        
        # Temporal reasoning
        "what happened", "what will happen", "what might happen",
        "before", "after", "next", "previously",
        
        # Spatial reasoning (complex)
        "relationship between", "relative to", "compared to",
        
        # Abstract concepts
        "mood", "emotion", "feeling", "atmosphere",
        "suggest", "imply", "indicate",
        
        # Multi-step reasoning
        "if", "would", "could", "should",
        "what would happen if", "what if",
        
        # Context & Background
        "context", "background", "story", "scenario",
        
        # Advice & Recommendations
        "recommend", "suggest", "advice", "should i",
        
        # Complex descriptions
        "describe in detail", "elaborate", "tell me more about",
    ]
    
    @staticmethod
    def classify(question):
        """
        Classify question complexity
        
        Args:
            question (str): The user's question
            
        Returns:
            str: "SIMPLE" or "COMPLEX"
        """
        question_lower = question.lower().strip()
        
        # Check for complex keywords first (higher priority)
        for keyword in QuestionComplexityClassifier.COMPLEX_KEYWORDS:
            if keyword in question_lower:
                return "COMPLEX"
        
        # Check for simple keywords
        for keyword in QuestionComplexityClassifier.SIMPLE_KEYWORDS:
            if keyword in question_lower:
                return "SIMPLE"
        
        # Default heuristics
        
        # Very short questions are usually simple
        if len(question_lower.split()) <= 5:
            return "SIMPLE"
        
        # Questions with multiple clauses are usually complex
        if question_lower.count(",") >= 2 or question_lower.count(" and ") >= 2:
            return "COMPLEX"
        
        # Questions ending with "why" or "how" are complex
        if question_lower.endswith("why") or question_lower.endswith("how"):
            return "COMPLEX"
        
        # Default to SIMPLE for unknown patterns
        # (BLIP will try first, then fallback to Gemini if needed)
        return "SIMPLE"
    
    @staticmethod
    def get_confidence(question):
        """
        Get confidence score for classification
        
        Returns:
            float: Confidence score 0.0-1.0
        """
        question_lower = question.lower().strip()
        
        # High confidence if explicit keywords found
        for keyword in QuestionComplexityClassifier.COMPLEX_KEYWORDS:
            if keyword in question_lower:
                return 0.9
        
        for keyword in QuestionComplexityClassifier.SIMPLE_KEYWORDS:
            if keyword in question_lower:
                return 0.9
        
        # Medium confidence for heuristics
        return 0.6
