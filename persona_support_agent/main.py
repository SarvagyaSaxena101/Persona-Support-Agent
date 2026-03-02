import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.services.orchestrator import SupportOrchestrator
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="AdsSparkX Persona API")

# Initialize Orchestrator
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("WARNING: GEMINI_API_KEY not found in environment. API will fail.")
orchestrator = SupportOrchestrator(gemini_api_key=api_key)

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        result = orchestrator.process_request(request.message)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
