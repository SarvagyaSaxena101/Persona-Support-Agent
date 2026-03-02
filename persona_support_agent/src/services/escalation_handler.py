import json
import logging
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EscalationHandler:
    """
    Evaluates conversations for high-risk triggers and provides a 
    structured handoff for human support agents.
    """

    def __init__(self):
        # Rule-based keyword triggers
        self.cancellation_keywords = ["cancel", "refund", "close account", "terminate", "stop subscription"]
        self.legal_keywords = ["sue", "lawyer", "legal", "attorney", "court", "compliance", "lawsuit"]

    def _check_keywords(self, message: str, keywords: List[str]) -> bool:
        """Helper to check if any keyword exists in the message."""
        message_lower = message.lower()
        return any(kw in message_lower for kw in keywords)

    def evaluate(
        self, 
        user_message: str, 
        persona: str, 
        confidence: float, 
        context_chunks: List[str]
    ) -> Dict[str, Any]:
        """
        Determines if a conversation should be escalated based on strict rules.
        Returns a structured handoff object.
        """
        escalation_flag = False
        reason = "No escalation triggered."

        # 1. Check for Cancellation requests
        if self._check_keywords(user_message, self.cancellation_keywords):
            escalation_flag = True
            reason = "User requested cancellation or refund."

        # 2. Check for Legal Action
        elif self._check_keywords(user_message, self.legal_keywords):
            escalation_flag = True
            reason = "User mentioned legal action or compliance issues."

        # 3. Check for Extreme Negative Sentiment (via Persona)
        elif persona == "Frustrated User" and confidence < 0.6:
            escalation_flag = True
            reason = "High frustration detected with low system confidence."

        # 4. Check for Low LLM Confidence
        elif confidence < 0.5:
            escalation_flag = True
            reason = f"System confidence score ({confidence}) below threshold."

        # 5. Check for Missing Context (Knowledge Base Gap)
        elif not context_chunks or len(context_chunks) == 0:
            escalation_flag = True
            reason = "No relevant knowledge base content found for this query."

        return self._generate_handoff(
            escalation_flag, 
            reason, 
            persona, 
            user_message, 
            context_chunks
        )

    def _generate_handoff(
        self, 
        should_escalate: bool, 
        reason: str, 
        persona: str, 
        message: str, 
        context: List[str]
    ) -> Dict[str, Any]:
        """
        Formats the output into a production-ready JSON schema for human agents.
        """
        return {
            "escalation": should_escalate,
            "reason": reason if should_escalate else None,
            "persona": persona,
            "conversation_summary": f"A {persona} is asking about: '{message[:50]}...'",
            "full_context": {
                "user_query": message,
                "retrieved_kb_snippets": context[:2],
                "metadata": {
                    "system_version": "1.0.0",
                    "priority": "HIGH" if should_escalate and ("legal" in reason.lower() or "cancel" in reason.lower()) else "NORMAL"
                }
            }
        }
