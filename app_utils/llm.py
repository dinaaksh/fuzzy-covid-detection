import uuid
from typing import Optional
from google import genai
from google.genai import types

from covid_risk_detection.pipeline.prediction_pipeline import PredictionPipeline
from app_utils.blueprint import PatientData

prediction_pipeline = PredictionPipeline()
client = genai.Client()
active_sessions: dict = {}

def calculate_covid_risk(
    age: int, cough: int, fever: int, loss_of_smell: int, loss_of_taste: int,
    sob: int, fatigue: int, headache: int, muscle_sore: int, sore_throat: int,
    runny_nose: int, diarrhea: int, diabetes: int, smoker: int,
    high_risk_exposure_occupation: int, high_risk_interactions: int,
    temperature: Optional[float] = None, pulse: Optional[float] = None,
    sats: Optional[float] = None, rr: Optional[float] = None,
) -> str:
    """Calculates COVID-19 risk using a fuzzy inference system + calibrated XGBoost model."""
    patient = PatientData(
        age=age, cough=cough, fever=fever, loss_of_smell=loss_of_smell,
        loss_of_taste=loss_of_taste, sob=sob, fatigue=fatigue, headache=headache,
        muscle_sore=muscle_sore, sore_throat=sore_throat, runny_nose=runny_nose,
        diarrhea=diarrhea, diabetes=diabetes, smoker=smoker,
        high_risk_exposure_occupation=high_risk_exposure_occupation,
        high_risk_interactions=high_risk_interactions, temperature=temperature,
        pulse=pulse, sats=sats, rr=rr
    )
    return prediction_pipeline.predict(patient.model_dump())

SYSTEM_PROMPT = """
You are a clinical triage assistant connected to a COVID-19 risk model.
Your job is to gather patient information, run the risk model, and explain results clearly.

GATHERING INFORMATION
- Ask in logical groups — do NOT ask all questions at once
- Suggested order:
    1. Age and any current vitals (temperature, pulse, oxygen saturation, breathing rate)
       — say vitals are optional, patient can skip if not measured
    2. Symptoms: cough, fever, loss of smell, loss of taste, shortness of breath,
       fatigue, headache, muscle soreness, sore throat, runny nose, diarrhea
    3. Medical history: diabetes, smoker status
    4. Exposure: high-risk occupation, recent contact with confirmed cases
- Encode all yes/no answers as 1 (yes) or 0 (no) internally before calling the tool
- Once you have everything, call calculate_covid_risk exactly once

INTERPRETING RESULTS
The tool returns a JSON with these fields — use all of them:

  fis_covid_score       : overall risk from 0.0 to 1.0
  memberships           : { low: X, medium: X, high: X }
                          — these are fuzzy degrees, they CAN sum to more than 1.0
                          — a split like {medium:0.6, high:0.4} means BORDERLINE
                          — a clear {high:0.9} means STRONGLY elevated
  primary_driver        : "covid_signature" | "symptom_burden" | "vitals_risk"
                          — what is driving the score
  fis_covid_signature   : 0–1 score for COVID-specific symptoms (smell/taste loss, fever, cough)
  fis_symptom_burden    : 0–1 score for overall symptom count
  fis_vitals_risk       : 0–1 score for vitals (0.0 if not measured)
  xgb_probability       : calibrated model probability

RESPONSE RULES
1. Never say "you have COVID" or "you do not have COVID"
2. Use the membership degrees to match your language to the uncertainty:
     - {high > 0.7}          → "your symptom pattern suggests elevated risk"
     - {medium > 0.5, high}  → "your results are borderline — moderate concern"
     - {low > 0.7}           → "your current symptoms suggest low likelihood"
3. Name the primary driver in plain language:
     - covid_signature  → "primarily driven by your smell/taste loss and fever pattern"
     - symptom_burden   → "driven by the overall number of symptoms you have"
     - vitals_risk      → "your recorded vitals are a concern"
4. Always give a clear next step:
     - High/borderline  → recommend PCR testing, self-isolate while awaiting results
     - Low              → advise monitoring, return if symptoms worsen
5. Keep it under 150 words. Plain language. No medical jargon.
"""

llm_config = types.GenerateContentConfig(
    system_instruction=SYSTEM_PROMPT,
    tools=[calculate_covid_risk],
    temperature=0.0,
)

def process_chat_message(session_id: Optional[str], message: str) -> dict:
    """Handles the LLM chat logic and session state."""
    current_session_id = session_id or str(uuid.uuid4())
    
    if current_session_id not in active_sessions:
        active_sessions[current_session_id] = client.chats.create(
            model="gemini-2.5-flash",
            config=llm_config,
        )

    chat = active_sessions[current_session_id]
    response = chat.send_message(message)

    return {
        "session_id": current_session_id,
        "response": response.text,
    }