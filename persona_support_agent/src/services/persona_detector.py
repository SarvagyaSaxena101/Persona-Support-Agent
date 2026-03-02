import json
import logging
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PersonaResponse(BaseModel):
    persona: str = Field(description="One of: Technical Expert, Frustrated User, Business Executive")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0")
    reasoning: str = Field(description="Short explanation of the classification")

class PersonaDetector:
    def __init__(self, api_key: str, model_name: str = "mistralai/mistral-7b-instruct", threshold: float = 0.7):
        # OpenRouter uses the OpenAI-compatible ChatOpenAI class
        self.llm = ChatOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            model=model_name,
            temperature=0
        )
        self.threshold = threshold
        self.parser = JsonOutputParser(pydantic_object=PersonaResponse)

    def detect_persona(self, message: str) -> Dict[str, Any]:
        prompt = ChatPromptTemplate.from_template("""
        You are a behavioral analyst. Classify this support query into EXACTLY one persona:
        1. Technical Expert: Focuses on implementation, API docs, error codes.
        2. Frustrated User: Emotional, high urgency, complaints.
        3. Business Executive: Focuses on ROI, SLAs, high-level features.

        User Message: {message}

        {format_instructions}
        """)

        chain = prompt | self.llm | self.parser

        try:
            result = chain.invoke({
                "message": message,
                "format_instructions": self.parser.get_format_instructions()
            })
            
            # Confidence Threshold Logic
            if result.get("confidence", 0) < self.threshold:
                result["original_persona"] = result["persona"]
                result["persona"] = "General User"
            
            return result
        except Exception as e:
            logger.error(f"Persona detection error: {e}")
            return {"persona": "General User", "confidence": 0, "reasoning": "Error in detection"}
