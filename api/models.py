from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime

# ==================== REQUEST MODELS ====================

class TextRecommendationRequest(BaseModel):
    """Request model for text-based meditation recommendations"""
    diary_text: str = Field(..., description="User's diary entry or current state description")
    user_id: str = Field(..., description="User ID for personalization")
    
    class Config:
        json_schema_extra = {
            "example": {
                "diary_text": "Feeling stressed about work deadlines and having trouble sleeping",
                "user_id": "user_123"
            }
        }

class ScriptGenerationRequest(BaseModel):
    """Request model for meditation script generation"""
    meditation_type: str = Field(..., description="Type of meditation to generate script for")
    user_id: str = Field(..., description="User ID for personalization")
    duration_minutes: int = Field(default=5, ge=1, le=30, description="Desired script duration in minutes")
    
    class Config:
        json_schema_extra = {
            "example": {
                "meditation_type": "Mindfulness Meditation",
                "user_id": "user_123",
                "duration_minutes": 8
            }
        }

class AudioAnalysisRequest(BaseModel):
    """Request model for audio analysis (used with file upload)"""
    user_id: str = Field(..., description="User ID")
    session_id: Optional[str] = Field(None, description="Optional session ID")
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user_123",
                "session_id": "session_456"
            }
        }

class FeedbackRequest(BaseModel):
    """Request model for user feedback"""
    user_id: str = Field(..., description="User ID")
    session_id: str = Field(..., description="Session ID being rated")
    meditation_type: str = Field(..., description="Type of meditation that was practiced")
    rating: int = Field(..., ge=1, le=5, description="Rating from 1-5 stars")
    comment: Optional[str] = Field(None, description="Optional feedback comment")
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user_123",
                "session_id": "session_456",
                "meditation_type": "Mindfulness Meditation",
                "rating": 4,
                "comment": "Really helpful for reducing anxiety"
            }
        }

class UserCreateRequest(BaseModel):
    """Request model for creating a new user"""
    name: str = Field(..., description="User's name")
    email: Optional[str] = Field(None, description="User's email")
    preferences: Optional[Dict[str, Any]] = Field(default_factory=dict, description="User preferences")
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "John Doe",
                "email": "john@example.com",
                "preferences": {
                    "experience_level": "beginner",
                    "preferred_duration": 10
                }
            }
        }

# ==================== RESPONSE MODELS ====================

class MeditationRecommendation(BaseModel):
    """Individual meditation recommendation"""
    meditation_type: str = Field(..., description="Name of the recommended meditation")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0.0-1.0)")
    rationale: str = Field(..., description="Explanation for why this meditation was recommended")
    source: str = Field(..., description="Source of recommendation (e.g., 'rule_based', 'ml_enhanced')")

class AudioInsights(BaseModel):
    """Audio analysis insights"""
    emotional_state: str = Field(..., description="Detected emotional state")
    stress_indicators: List[str] = Field(..., description="List of detected stress indicators")
    voice_energy: str = Field(..., description="Voice energy level (low/moderate/high)")

class RecommendationResponse(BaseModel):
    """Response model for meditation recommendations"""
    session_id: str = Field(..., description="Session ID for tracking")
    recommendations: List[MeditationRecommendation] = Field(..., description="List of meditation recommendations")
    status: str = Field(..., description="Processing status")
    method: str = Field(..., description="Method used for recommendations")
    audio_insights: Optional[AudioInsights] = Field(None, description="Audio analysis insights if audio was processed")
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "session_456",
                "recommendations": [
                    {
                        "meditation_type": "Breathing Meditation",
                        "confidence": 0.92,
                        "rationale": "Recommended for stress and anxiety relief",
                        "source": "rule_based"
                    }
                ],
                "status": "completed",
                "method": "text_analysis"
            }
        }

class ScriptResponse(BaseModel):
    """Response model for generated meditation scripts"""
    meditation_type: str = Field(..., description="Type of meditation")
    instructions: str = Field(..., description="Basic meditation instructions")
    script: str = Field(..., description="Complete TTS-ready meditation script")
    duration_minutes: str = Field(..., description="Expected duration when spoken")
    format: str = Field(..., description="Script format type")
    
    class Config:
        json_schema_extra = {
            "example": {
                "meditation_type": "Mindfulness Meditation",
                "instructions": "Focus on present moment awareness",
                "script": "Welcome to your mindfulness practice. Find a comfortable position...",
                "duration_minutes": "5-7",
                "format": "TTS-ready"
            }
        }

class AudioProcessingResponse(BaseModel):
    """Response model for audio processing requests"""
    session_id: str = Field(..., description="Session ID for tracking")
    audio_url: str = Field(..., description="URL of uploaded audio file")
    status: str = Field(..., description="Processing status")
    message: str = Field(..., description="Status message")
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "session_456",
                "audio_url": "https://storage.googleapis.com/bucket/audio/session_456.wav",
                "status": "processing",
                "message": "Audio uploaded successfully. Processing in background."
            }
        }

class SessionStatusResponse(BaseModel):
    """Response model for session status queries"""
    session_id: str = Field(..., description="Session ID")
    status: str = Field(..., description="Current status (processing/completed/failed)")
    created_at: Optional[datetime] = Field(None, description="Session creation time")
    has_results: bool = Field(..., description="Whether results are available")
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "session_456",
                "status": "completed",
                "created_at": "2024-01-01T10:00:00Z",
                "has_results": True
            }
        }

class SessionResultsResponse(BaseModel):
    """Response model for session results"""
    session_id: str = Field(..., description="Session ID")
    status: str = Field(..., description="Processing status")
    results: Dict[str, Any] = Field(..., description="Processing results")
    audio_url: Optional[str] = Field(None, description="URL of processed audio file")
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "session_456",
                "status": "completed",
                "results": {
                    "audio_analysis": {
                        "features": {...},
                        "embeddings": {...}
                    }
                },
                "audio_url": "https://storage.googleapis.com/bucket/audio/session_456.wav"
            }
        }

class FeedbackResponse(BaseModel):
    """Response model for feedback submission"""
    feedback_id: str = Field(..., description="Feedback record ID")
    message: str = Field(..., description="Confirmation message")
    
    class Config:
        json_schema_extra = {
            "example": {
                "feedback_id": "feedback_789",
                "message": "Feedback recorded successfully"
            }
        }

class UserResponse(BaseModel):
    """Response model for user data"""
    user_id: str = Field(..., description="User ID")
    name: str = Field(..., description="User name")
    email: Optional[str] = Field(None, description="User email")
    preferences: Dict[str, Any] = Field(..., description="User preferences")
    created_at: Optional[datetime] = Field(None, description="Account creation time")
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user_123",
                "name": "John Doe",
                "email": "john@example.com",
                "preferences": {
                    "experience_level": "beginner",
                    "favorite_meditations": ["mindfulness", "breathing"]
                },
                "created_at": "2024-01-01T10:00:00Z"
            }
        }

class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str = Field(..., description="API status")
    timestamp: datetime = Field(..., description="Current timestamp")
    version: str = Field(..., description="API version")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2024-01-01T10:00:00Z",
                "version": "1.0.0"
            }
        }

class ErrorResponse(BaseModel):
    """Response model for API errors"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "validation_error",
                "message": "Invalid input data",
                "detail": "diary_text field is required"
            }
        }