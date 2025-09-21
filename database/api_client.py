import aiohttp
import asyncio
from typing import Dict, Any, List, Optional, Union
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class MeditationDBAPIClient:
    """
    Client for MeditationDB API
    Handles all HTTP requests to the external API endpoint
    """
    
    def __init__(self, base_url: str = "https://meditationdb-api-222233295505.asia-south1.run.app"):
        self.base_url = base_url.rstrip('/')
        self.api_base = f"{self.base_url}/api"
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session
    
    async def close(self):
        """Close the HTTP session"""
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def _make_request(
        self, 
        method: str, 
        endpoint: str, 
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make HTTP request to the API
        
        Args:
            method: HTTP method (GET, POST, PUT, PATCH, DELETE)
            endpoint: API endpoint (without /api prefix)
            data: Request body data
            params: Query parameters
            
        Returns:
            API response data
        """
        session = await self._get_session()
        url = f"{self.api_base}/{endpoint.lstrip('/')}"
        
        try:
            headers = {"Content-Type": "application/json"} if data else {}
            
            async with session.request(
                method=method,
                url=url,
                json=data,
                params=params,
                headers=headers
            ) as response:
                
                # Handle different response status codes
                if response.status == 200:
                    result = await response.json()
                    return result.get('data', result) if 'data' in result else result
                elif response.status == 201:
                    result = await response.json()
                    return result.get('data', result) if 'data' in result else result
                elif response.status == 404:
                    return None
                elif response.status >= 400:
                    error_text = await response.text()
                    logger.error(f"API Error {response.status}: {error_text}")
                    raise Exception(f"API request failed with status {response.status}: {error_text}")
                
                return await response.json()
                
        except aiohttp.ClientError as e:
            logger.error(f"HTTP client error: {e}")
            raise Exception(f"Network error: {str(e)}")
        except Exception as e:
            logger.error(f"Request error: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    # ==================== USER OPERATIONS ====================
    
    async def create_user(self, user_data: Dict[str, Any]) -> str:
        """
        Create a new user
        
        Args:
            user_data: User data containing name, email, preferences
            
        Returns:
            User ID
        """
        # Map our user data to API schema
        api_data = {
            "name": user_data.get("name", ""),
            "email": user_data.get("email"),
            "preferences": user_data.get("preferences", {})
        }
        
        # Ensure preferences have required fields for the API
        if "preferences" not in api_data or not api_data["preferences"]:
            api_data["preferences"] = {}
        
        # Set default experience level if not provided
        if "experience_level" not in api_data["preferences"]:
            api_data["preferences"]["experience_level"] = "beginner"
        
        # Set default goals if not provided (using valid API values)
        if "goals" not in api_data["preferences"]:
            api_data["preferences"]["goals"] = ["stress_reduction"]
            
        result = await self._make_request("POST", "/users", api_data)
        return result.get("id") or result.get("_id")
    
    async def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by ID"""
        result = await self._make_request("GET", f"/users/{user_id}")
        return result
    
    async def update_user(self, user_id: str, updates: Dict[str, Any]) -> bool:
        """Update user data"""
        result = await self._make_request("PUT", f"/users/{user_id}", updates)
        return result is not None
    
    async def update_user_preferences(self, user_id: str, preferences: Dict[str, Any]) -> bool:
        """Update user preferences only"""
        result = await self._make_request("PATCH", f"/users/{user_id}/preferences", {"preferences": preferences})
        return result is not None
    
    async def get_users(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """Get all users with pagination"""
        params = {"limit": limit, "offset": offset}
        result = await self._make_request("GET", "/users", params=params)
        return result if isinstance(result, list) else result.get("users", [])
    
    # ==================== SESSION OPERATIONS ====================
    
    async def create_session(self, session_data: Dict[str, Any]) -> str:
        """
        Create a meditation session
        
        Args:
            session_data: Session data
            
        Returns:
            Session ID
        """
        # Map our session data to API schema - only include required fields
        api_data = {
            "user_id": session_data.get("user_id"),
            "input_type": session_data.get("input_type", "text"),
            "input_data": session_data.get("input_data", {})
        }
        
        result = await self._make_request("POST", "/sessions", api_data)
        return result.get("id") or result.get("_id")
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session by ID"""
        return await self._make_request("GET", f"/sessions/{session_id}")
    
    async def update_session(self, session_id: str, updates: Dict[str, Any]) -> bool:
        """Update session data"""
        result = await self._make_request("PUT", f"/sessions/{session_id}", updates)
        return result is not None
    
    async def update_session_status(self, session_id: str, status: str) -> bool:
        """Update session status only"""
        result = await self._make_request("PATCH", f"/sessions/{session_id}/status", {"status": status})
        return result is not None
    
    async def update_session_results(self, session_id: str, results: Dict[str, Any]) -> bool:
        """Update session results"""
        result = await self._make_request("PATCH", f"/sessions/{session_id}/results", {"results": results})
        return result is not None
    
    async def get_user_sessions(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get user's sessions"""
        result = await self._make_request("GET", f"/sessions/user/{user_id}", params={"limit": limit})
        return result if isinstance(result, list) else result.get("sessions", [])
    
    async def get_sessions(self, status: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Get sessions with optional status filter"""
        params = {"limit": limit}
        if status:
            params["status"] = status
        result = await self._make_request("GET", "/sessions", params=params)
        return result if isinstance(result, list) else result.get("sessions", [])
    
    # ==================== FEEDBACK OPERATIONS ====================
    
    async def create_feedback(self, feedback_data: Dict[str, Any]) -> str:
        """Create feedback"""
        api_data = {
            "user_id": feedback_data.get("user_id"),
            "session_id": feedback_data.get("session_id"),
            "meditation_type": feedback_data.get("meditation_type", "mindfulness"),
            "rating": feedback_data.get("rating")
        }
        
        result = await self._make_request("POST", "/feedback", api_data)
        return result.get("id") or result.get("_id")
    
    async def get_feedback(self, feedback_id: str) -> Optional[Dict[str, Any]]:
        """Get feedback by ID"""
        return await self._make_request("GET", f"/feedback/{feedback_id}")
    
    async def get_user_feedback(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get user's feedback"""
        result = await self._make_request("GET", f"/feedback/user/{user_id}", params={"limit": limit})
        return result if isinstance(result, list) else result.get("feedback", [])
    
    async def get_session_feedback(self, session_id: str) -> List[Dict[str, Any]]:
        """Get feedback for a specific session"""
        result = await self._make_request("GET", f"/feedback/session/{session_id}")
        return result if isinstance(result, list) else result.get("feedback", [])
    
    # ==================== VECTOR OPERATIONS ====================
    
    async def create_vector(self, vector_data: Dict[str, Any]) -> str:
        """Create vector embedding"""
        api_data = {
            "entity_id": vector_data.get("entity_id"),
            "entity_type": vector_data.get("entity_type", "user"),
            "embedding": vector_data.get("embedding", []),
            "metadata": vector_data.get("metadata", {})
        }
        
        result = await self._make_request("POST", "/vectors", api_data)
        return result.get("id") or result.get("_id")
    
    async def get_vector(self, vector_id: str) -> Optional[Dict[str, Any]]:
        """Get vector by ID"""
        return await self._make_request("GET", f"/vectors/{vector_id}")
    
    async def vector_similarity_search(
        self, 
        query_vector: List[float], 
        entity_type: str = "user", 
        limit: int = 10,
        min_similarity: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Perform vector similarity search"""
        api_data = {
            "query_vector": query_vector,
            "entity_type": entity_type,
            "limit": limit,
            "min_similarity": min_similarity
        }
        
        result = await self._make_request("POST", "/vectors/similarity", api_data)
        return result if isinstance(result, list) else result.get("results", [])
    
    async def get_meditation_recommendations(
        self, 
        user_id: str, 
        user_preferences: Optional[Dict[str, Any]] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Get AI-powered meditation recommendations"""
        api_data = {
            "user_id": user_id,
            "preferences": user_preferences or {},
            "limit": limit
        }
        
        result = await self._make_request("POST", "/vectors/meditation-recommendations", api_data)
        return result if isinstance(result, list) else result.get("recommendations", [])
    
    # ==================== HISTORY OPERATIONS ====================
    
    async def create_history_entry(self, history_data: Dict[str, Any]) -> str:
        """Create history entry"""
        api_data = {
            "user_id": history_data.get("user_id"),
            "session_id": history_data.get("session_id"),
            "meditation_type": history_data.get("meditation_type"),
            "duration": history_data.get("duration"),
            "rating": history_data.get("rating"),
            "notes": history_data.get("notes", ""),
            "metadata": history_data.get("metadata", {})
        }
        
        result = await self._make_request("POST", "/history", api_data)
        return result.get("id") or result.get("_id")
    
    async def get_user_history(self, user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get user's meditation history"""
        result = await self._make_request("GET", f"/history/user/{user_id}", params={"limit": limit})
        return result if isinstance(result, list) else result.get("history", [])
    
    async def get_user_progress(self, user_id: str) -> Dict[str, Any]:
        """Get user's progress analytics"""
        result = await self._make_request("GET", f"/history/user/{user_id}/progress")
        return result or {}
    
    # ==================== ANALYTICS OPERATIONS ====================
    
    async def get_analytics_overview(self, timeframe: str = "30d") -> Dict[str, Any]:
        """Get analytics overview"""
        result = await self._make_request("GET", "/analytics/overview", params={"timeframe": timeframe})
        return result or {}
    
    async def get_user_analytics(self, user_id: str) -> Dict[str, Any]:
        """Get user-specific analytics"""
        result = await self._make_request("GET", "/analytics/users", params={"user_id": user_id})
        return result or {}
    
    # ==================== HEALTH CHECK ====================
    
    async def health_check(self) -> Dict[str, Any]:
        """Check API health status"""
        return await self._make_request("GET", "/health")
    
    async def get_api_info(self) -> Dict[str, Any]:
        """Get API information"""
        return await self._make_request("GET", "/info")


# Global client instance
_api_client = None

def get_api_client() -> MeditationDBAPIClient:
    """Get or create the API client instance"""
    global _api_client
    if _api_client is None:
        _api_client = MeditationDBAPIClient()
    return _api_client

async def close_api_client():
    """Close the API client connection"""
    global _api_client
    if _api_client:
        await _api_client.close()
        _api_client = None