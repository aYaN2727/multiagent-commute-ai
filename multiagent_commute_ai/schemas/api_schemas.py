"""
schemas/api_schemas.py
Pydantic v2 request/response models for the FastAPI layer.
"""
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict


class CommuteRecord(BaseModel):
    """Structured representation of a single commute event."""
    route_id: str = "UNKNOWN"
    distance_km: float = 0.0
    delay_minutes: float = 0.0
    route_avg_delay_min: float = 10.0
    day_of_week: int = 0         # 0 = Monday, 6 = Sunday
    hour_of_day: int = 9         # 24-hour format
    claim_frequency_30d: float = 0.0
    week_num: int = 1
    is_holiday: int = 0


class ChatMessage(BaseModel):
    """A single turn in the conversation history."""
    role: str    # "user" or "assistant"
    content: str


class QueryRequest(BaseModel):
    """Incoming request for a policy/claim query."""
    employee_id: str
    query: str
    commute_record: Optional[CommuteRecord] = None
    # Last N conversation turns sent from the client for context continuity.
    # Max 6 messages (3 full turns) are used; older history is ignored.
    conversation_history: List[ChatMessage] = []

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "employee_id": "EMP_4821",
                "query": "My bus was 90 minutes late today. Am I eligible for cab reimbursement?",
                "commute_record": {
                    "route_id": "ROUTE_07",
                    "distance_km": 14.5,
                    "delay_minutes": 90,
                    "route_avg_delay_min": 8,
                    "day_of_week": 1,
                    "hour_of_day": 9,
                    "claim_frequency_30d": 4,
                    "week_num": 12,
                    "is_holiday": 0,
                },
            }
        }
    )


class QueryResponse(BaseModel):
    """Full structured response returned to the client."""
    employee_id: str
    intent: str
    final_response: str
    policy_answer: Optional[str] = None
    source_sections: List[str] = []
    anomaly_flagged: bool = False
    anomaly_score: Optional[float] = None
    anomaly_probability: Optional[float] = None
    shap_explanation: List[str] = []
    overall_confidence: float
    needs_escalation: bool
    escalation_reason: Optional[str] = None
    processing_errors: List[str] = []


class HealthResponse(BaseModel):
    """API health check response."""
    status: str
    models_loaded: Dict[str, bool]
    version: str = "1.0.0"


class ErrorResponse(BaseModel):
    """Structured error response."""
    error: str
    detail: str
