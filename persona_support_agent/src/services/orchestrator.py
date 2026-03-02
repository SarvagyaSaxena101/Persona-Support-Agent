from typing import List, Dict, Any
from .persona_detector import PersonaDetector
from ..rag.vector_store import VectorStore
from .response_generator import ResponseGenerator
from .escalation_handler import EscalationHandler
from ..utils.analytics_logger import log_persona_event

class SupportOrchestrator:
    def __init__(self, api_key: str):
        self.detector = PersonaDetector(api_key=api_key)
        self.vector_store = VectorStore()
        self.generator = ResponseGenerator(api_key=api_key)
        self.escalator = EscalationHandler()
        
        # --- Conversation Memory ---
        self.memory_limit = 5
        self.history: List[Dict[str, str]] = []

    def process_request(self, user_message: str) -> Dict[str, Any]:
        # 1. Detect Persona
        persona_data = self.detector.detect_persona(user_message)
        persona = persona_data["persona"]
        
        # 2. Log Analytics
        log_persona_event(persona, persona_data["confidence"], len(user_message))

        # 3. Retrieve Context (RAG)
        context_chunks = self.vector_store.search(user_message)

        # 4. Generate Response
        response_data = self.generator.generate_response(user_message, persona, context_chunks)

        # 5. Evaluate Escalation
        escalation_data = self.escalator.evaluate(
            user_message, persona, response_data["confidence"], context_chunks
        )

        # 6. Update Memory
        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": response_data["response"]})
        if len(self.history) > (self.memory_limit * 2):
            self.history = self.history[-(self.memory_limit * 2):]

        return {
            "response": response_data["response"],
            "persona_info": persona_data,
            "response_info": response_data,
            "escalation": escalation_data
        }
