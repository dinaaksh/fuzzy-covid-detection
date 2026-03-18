import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from app_utils.blueprint import PatientData, ChatMessage
from app_utils.llm import prediction_pipeline, process_chat_message

load_dotenv()

app = FastAPI(title="COVID-19 Risk Assessment API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/predict")
def predict_direct(data: PatientData):
    """Direct, non-conversational endpoint for the predictive model."""
    try:
        result_string = prediction_pipeline.predict(data.model_dump())
        return json.loads(result_string)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
def chat_with_llm(chat_request: ChatMessage):
    """Conversational endpoint powered by Gemini and the Risk Model Tool."""
    try:
        return process_chat_message(chat_request.session_id, chat_request.message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))