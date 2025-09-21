"""
MeditationDB API Constants and Valid Values
"""

# Valid user experience levels
EXPERIENCE_LEVELS = [
    "beginner",
    "intermediate", 
    "advanced"
]

# Valid meditation goals - must match API validation
VALID_GOALS = [
    "stress_reduction",
    "better_sleep",
    "focus",
    "emotional_balance",
    "anxiety_relief",
    "pain_management",
    "spiritual_growth",
    "creativity",
    "self_awareness",
    "compassion",
    "patience",
    "mindful_eating"
]

# Common meditation types - must match API validation
MEDITATION_TYPES = [
    "mindfulness",
    "breathing", 
    "body_scan",
    "loving_kindness",
    "walking",
    "guided_imagery",
    "mantra",
    "zen",
    "vipassana",
    "transcendental",
    "movement",
    "sound_bath"
]

# Input types for sessions
SESSION_INPUT_TYPES = [
    "audio",
    "text"
]

# Session statuses
SESSION_STATUSES = [
    "pending",
    "processing", 
    "completed",
    "failed"
]

# Vector entity types
VECTOR_ENTITY_TYPES = [
    "user",
    "session",
    "meditation",
    "audio"
]

def validate_user_preferences(preferences: dict) -> dict:
    """
    Validate and sanitize user preferences according to API requirements
    
    Args:
        preferences: User preferences dict
        
    Returns:
        Validated preferences dict
    """
    validated = {}
    
    # Experience level
    experience_level = preferences.get("experience_level", "beginner")
    if experience_level not in EXPERIENCE_LEVELS:
        experience_level = "beginner"
    validated["experience_level"] = experience_level
    
    # Preferred duration
    duration = preferences.get("preferred_duration", 10)
    if not isinstance(duration, int) or duration < 1:
        duration = 10
    validated["preferred_duration"] = duration
    
    # Favorite meditations
    favorite_meditations = preferences.get("favorite_meditations", [])
    if not isinstance(favorite_meditations, list):
        favorite_meditations = []
    validated["favorite_meditations"] = favorite_meditations
    
    # Goals - validate against allowed values
    goals = preferences.get("goals", [])
    if not isinstance(goals, list):
        goals = ["stress_reduction"]
    else:
        # Filter out invalid goals
        valid_goals = [goal for goal in goals if goal in VALID_GOALS]
        if not valid_goals:
            valid_goals = ["stress_reduction"]
        goals = valid_goals
    validated["goals"] = goals
    
    return validated

def get_default_user_preferences() -> dict:
    """Get default user preferences that pass API validation"""
    return {
        "experience_level": "beginner",
        "preferred_duration": 10,
        "favorite_meditations": ["mindfulness"],
        "goals": ["stress_reduction"]
    }

def create_test_user_data(name: str = "Test User", email: str = "test@example.com") -> dict:
    """Create valid test user data"""
    return {
        "name": name,
        "email": email,
        "preferences": get_default_user_preferences()
    }

def map_meditation_type_to_api(internal_type: str) -> str:
    """
    Map internal meditation types to API-compatible types
    
    Args:
        internal_type: Internal meditation type name
        
    Returns:
        API-compatible meditation type
    """
    # Mapping from internal types to API types
    type_mapping = {
        "Mindfulness Meditation": "mindfulness",
        "Body Scan Meditation": "body_scan", 
        "Breathing Meditation": "breathing",
        "Loving-Kindness Meditation": "loving_kindness",
        "Progressive Muscle Relaxation": "movement",
        "Visualization Meditation": "guided_imagery",
        "Walking Meditation": "walking",
        "Mantra Meditation": "mantra",
        "Zen Meditation": "zen",
        "Vipassana Meditation": "vipassana",
        "Transcendental Meditation": "transcendental",
        "Sound Bath": "sound_bath",
        
        # Handle lowercase versions
        "mindfulness meditation": "mindfulness",
        "body scan meditation": "body_scan",
        "breathing meditation": "breathing",
        "loving-kindness meditation": "loving_kindness", 
        "progressive muscle relaxation": "movement",
        "visualization meditation": "guided_imagery",
        "walking meditation": "walking",
        "mantra meditation": "mantra",
        "zen meditation": "zen",
        "vipassana meditation": "vipassana",
        "transcendental meditation": "transcendental",
        "sound bath": "sound_bath",
        
        # Direct matches (already correct)
        "mindfulness": "mindfulness",
        "breathing": "breathing", 
        "body_scan": "body_scan",
        "loving_kindness": "loving_kindness",
        "walking": "walking",
        "guided_imagery": "guided_imagery",
        "mantra": "mantra",
        "zen": "zen",
        "vipassana": "vipassana",
        "transcendental": "transcendental",
        "movement": "movement",
        "sound_bath": "sound_bath"
    }
    
    # Return mapped type or default to mindfulness if not found
    return type_mapping.get(internal_type.lower(), "mindfulness")

def validate_meditation_recommendations(recommendations: list) -> list:
    """
    Validate and fix meditation recommendations to match API requirements
    
    Args:
        recommendations: List of recommendation dictionaries
        
    Returns:
        List of validated recommendations
    """
    # Valid source values according to API
    valid_sources = ["text_analysis", "audio_analysis", "user_history", "similarity_match"]
    
    validated = []
    
    for rec in recommendations:
        if isinstance(rec, dict):
            # Map the meditation type to API-compatible format
            if 'meditation_type' in rec:
                rec['meditation_type'] = map_meditation_type_to_api(rec['meditation_type'])
            elif 'type' in rec:
                rec['meditation_type'] = map_meditation_type_to_api(rec['type'])
                # Remove the old 'type' field
                rec.pop('type', None)
            else:
                # Add default meditation type if missing
                rec['meditation_type'] = "mindfulness"
            
            # Fix source field if present and invalid
            if 'source' in rec and rec['source'] not in valid_sources:
                # Map common internal source values to API values
                source_mapping = {
                    'rule_base_disorder': 'text_analysis',
                    'ml_prediction': 'text_analysis', 
                    'fallback': 'text_analysis',
                    'default': 'text_analysis'
                }
                rec['source'] = source_mapping.get(rec['source'], 'text_analysis')
            elif 'source' not in rec:
                # Add default source if missing
                rec['source'] = 'text_analysis'
            
            validated.append(rec)
    
    return validated

def sanitize_audio_analysis_results(audio_results: dict) -> dict:
    """
    Remove fields from audio analysis results that are not allowed by the API
    
    The API has strict validation and doesn't accept complex nested feature structures.
    This function creates a simplified version with only basic fields.
    
    Args:
        audio_results: Raw audio analysis results
        
    Returns:
        Sanitized results with only allowed fields
    """
    if not isinstance(audio_results, dict):
        return {}
    
    # Create a simplified structure that the API will accept
    sanitized = {}
    
    # Keep basic fields
    if 'file' in audio_results:
        sanitized['file'] = audio_results['file']
    
    if 'audio_embedding' in audio_results:
        # Keep embedding but limit size to prevent payload issues
        embedding = audio_results['audio_embedding']
        if isinstance(embedding, list) and len(embedding) > 100:
            # Truncate very large embeddings
            sanitized['audio_embedding'] = embedding[:100]
        else:
            sanitized['audio_embedding'] = embedding
    
    # Extract key metrics from nested features into flat structure
    features = audio_results.get('features', {})
    
    # Extract emotion info
    if 'emotion' in features:
        emotion = features['emotion']
        if isinstance(emotion, dict):
            sanitized['emotion_label'] = emotion.get('label', 'neutral')
            if 'probs' in emotion and isinstance(emotion['probs'], list):
                sanitized['emotion_confidence'] = max(emotion['probs']) if emotion['probs'] else 0.0
    
    # Extract basic VAD info without complex nested structure
    if 'vad' in features:
        vad = features['vad']
        if isinstance(vad, dict):
            sanitized['speech_ratio'] = vad.get('speech_ratio', 0.0)
            sanitized['segment_count'] = vad.get('segment_count', 0)
    
    # Add processing status
    sanitized['processing_status'] = 'completed'
    
    return sanitized