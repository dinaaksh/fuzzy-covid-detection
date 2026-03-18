from typing import Optional
from pydantic import BaseModel

class PatientData(BaseModel):
    age: int
    cough: int
    fever: int
    loss_of_smell: int
    loss_of_taste: int
    sob: int
    fatigue: int
    headache: int
    muscle_sore: int
    sore_throat: int
    runny_nose: int
    diarrhea: int
    diabetes: int
    smoker: int
    high_risk_exposure_occupation: int
    high_risk_interactions: int
    
    temperature: Optional[float] = None
    pulse: Optional[float] = None
    sats: Optional[float] = None
    rr: Optional[float] = None

class ChatMessage(BaseModel):
    session_id: Optional[str] = None
    message: str