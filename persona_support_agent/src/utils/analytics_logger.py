import logging
import json
from datetime import datetime
import os

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Set up a specific logger for persona analytics
persona_logger = logging.getLogger("persona_analytics")
handler = logging.FileHandler("logs/persona_distribution.log")
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
persona_logger.addHandler(handler)
persona_logger.setLevel(logging.INFO)

def log_persona_event(persona: str, confidence: float, message_length: int):
    """Logs persona detection events in a structured format for later analysis."""
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "persona": persona,
        "confidence": round(confidence, 2),
        "message_length": message_length
    }
    persona_logger.info(json.dumps(log_entry))
