import json
import logging
from typing import Dict, List, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SupportResponse(BaseModel):
    response: str = Field(description="The response message to the user.")
    confidence: float = Field(description="Confidence score for the answer provided.")

class ResponseGenerator:
    def __init__(self, api_key: str, model_name: str = "mistralai/mistral-7b-instruct"):
        self.llm = ChatOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            model=model_name,
            temperature=0.3
        )
        self.parser = JsonOutputParser(pydantic_object=SupportResponse)

    def _get_persona_instructions(self, persona: str) -> str:
        instructions = {
            "Technical Expert": (
                "Role: Senior Systems Engineer. Tone: Objective, precise, and technical. "
                "Constraint: Mention specific details and avoid fluff."
            ),
            "Frustrated User": (
                "Role: Senior Support Specialist. Tone: Deeply empathetic and apologetic. "
                "Constraint: Acknowledge frustration and provide a step-by-step resolution."
            ),
            "Business Executive": (
                "Role: Strategic Account Manager. Tone: Professional, high-level, results-oriented. "
                "Constraint: Use bullet points and focus on ROI/Scale."
            )
        }
        return instructions.get(persona, "Role: Helpful Support Assistant. Tone: Professional and balanced.")

    def generate_response(self, user_message: str, persona: str, context_chunks: List[str]) -> Dict[str, Any]:
        persona_style = self._get_persona_instructions(persona)
        context_text = "\n---\n".join(context_chunks) if context_chunks else "No specific context found."

        prompt = ChatPromptTemplate.from_template("""
        You are a Persona-Adaptive Support Agent for AdsSparkX.
        ADAPTIVE STYLE: {style}
        CONTEXT: {context}

        Answer the user's question accurately using only the provided context. 
        If not in context, state that you need to escalate to a human.
        USER MESSAGE: {message}

        {format_instructions}
        """)

        chain = prompt | self.llm | self.parser

        try:
            return chain.invoke({
                "style": persona_style,
                "context": context_text,
                "message": user_message,
                "format_instructions": self.parser.get_format_instructions()
            })
        except Exception as e:
            logger.error(f"Response generation error: {e}")
            return {"response": "I apologize, but I am unable to process your request at this time.", "confidence": 0}
