from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import json
import uuid
import os
from dotenv import load_dotenv

load_dotenv()

from covid_risk_detection.pipeline.prediction_pipeline import PredictionPipeline

from google import genai
from google.genai import types

app = FastAPI(title="COVID-19 Risk Assessment API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

prediction_pipeline = PredictionPipeline()

class PatientData(BaseModel):
    age: int; temperature: float; pulse: float; sats: float; rr: float; duration: int
    cough: int; fever: int; loss_of_smell: int; loss_of_taste: int; sob: int; fatigue: int
    headache: int; muscle_sore: int; sore_throat: int; runny_nose: int; diarrhea: int
    cough_severity: int; sob_severity: int
    diabetes: int; chd: int; htn: int; cancer: int; asthma: int; copd: int; autoimmune_dis: int; smoker: int
    high_risk_exposure_occupation: int; high_risk_interactions: int
    flu_negative: int; flu_positive: int; flu_tested: int

@app.post("/api/predict")
def predict_direct(data: PatientData):
    try:
        raw_data = data.model_dump()
        result_string = prediction_pipeline.predict(raw_data)
        return json.loads(result_string)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

client = genai.Client()

def calculate_covid_risk(
    age: int, temperature: float, pulse: float, sats: float, rr: float, duration: int,
    cough: int, fever: int, loss_of_smell: int, loss_of_taste: int, sob: int, fatigue: int,
    headache: int, muscle_sore: int, sore_throat: int, runny_nose: int, diarrhea: int,
    cough_severity: int, sob_severity: int,
    diabetes: int, chd: int, htn: int, cancer: int, asthma: int, copd: int, autoimmune_dis: int, smoker: int,
    high_risk_exposure_occupation: int, high_risk_interactions: int,
    flu_negative: int, flu_positive: int, flu_tested: int
) -> str:
    """Calculates COVID-19 risk probability based on patient vitals, symptoms, and medical history. Call this only when all information is gathered."""
    raw_data = locals()
    return prediction_pipeline.predict(raw_data)

system_prompt = """
You are a clinical triage assistant. Gather patient data to run a COVID-19 risk model.
Ask for vitals, demographics, symptom presence, symptom severity, medical history, and exposure/testing.
Group questions logically. Do not ask for all 32 inputs at once.
Once you have EVERYTHING, call the calculate_covid_risk tool and explain the probability clearly.
"""

config = types.GenerateContentConfig(
    system_instruction=system_prompt,
    tools=[calculate_covid_risk],
    temperature=0.0,
)

active_sessions = {}

class ChatMessage(BaseModel):
    session_id: str | None = None
    message: str

@app.post("/api/chat")
def chat_with_llm(chat_request: ChatMessage):
    try:
        session_id = chat_request.session_id
        if not session_id or session_id not in active_sessions:
            session_id = str(uuid.uuid4())
            active_sessions[session_id] = client.chats.create(
                model="gemini-2.5-flash", 
                config=config
            )
            
        chat = active_sessions[session_id]
        response = chat.send_message(chat_request.message)
        
        return {
            "session_id": session_id,
            "response": response.text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))