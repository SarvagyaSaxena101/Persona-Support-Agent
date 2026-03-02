import streamlit as st
import os
from src.services.orchestrator import SupportOrchestrator
from src.rag.vector_store import VectorStore

# --- Configuration ---
st.set_page_config(page_title="AdsSparkX AI Support", layout="wide", page_icon="💬")

# --- Custom Styling (Vanilla CSS) ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stChatFloatingInputContainer { background-color: white; border-top: 1px solid #ddd; }
    .st-emotion-cache-1c7n2ka { background-color: #e3f2fd; border-radius: 15px; padding: 15px; border-bottom-left-radius: 0; }
    .st-emotion-cache-10pw50 { background-color: #ffffff; border-radius: 15px; padding: 15px; border-bottom-right-radius: 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar: Configuration & Diagnostics ---
with st.sidebar:
    st.title("🛡️ System Settings")
    
    # 1. API Key Injection
    openrouter_key = st.text_input("OpenRouter API Key", type="password", help="Enter your OpenRouter API key.")
    
    # 2. Model Selection (Free Models)
    model_choice = st.selectbox(
        "Select Model",
        [
            "mistralai/mistral-7b-instruct",
            "google/gemma-2-9b-it",
            "meta-llama/llama-3-8b-instruct",
            "microsoft/phi-3-mini-128k-instruct"
        ],
        help="Choose a free or low-cost model from OpenRouter."
    )
    
    if not openrouter_key:
        st.warning("⚠️ Please provide an API key to start.")
    
    st.divider()
    
    # 3. Rebuild Knowledge Base Index
    if st.button("🔄 Rebuild Knowledge Base"):
        with st.spinner("Indexing knowledge base..."):
            try:
                vs = VectorStore()
                vs.build_index()
                st.success("Indexing complete!")
            except Exception as e:
                st.error(f"Indexing failed: {e}")
    
    st.divider()

    # 4. Diagnostics
    if "last_metadata" not in st.session_state:
        st.session_state.last_metadata = {"persona": "Not Detected", "confidence": 0.0, "escalated": False, "reason": "None"}
    
    st.subheader("Detected Persona")
    persona = st.session_state.last_metadata["persona"]
    p_color = "blue" if persona == "Technical Expert" else "orange" if persona == "Business Executive" else "red"
    st.markdown(f"**Status:** :{p_color}[{persona}]")
    
    st.subheader("AI Confidence")
    st.progress(st.session_state.last_metadata["confidence"])
    
    st.subheader("Human Handoff")
    if st.session_state.last_metadata["escalated"]:
        st.error(f"⚠️ Escalated: {st.session_state.last_metadata['reason']}")
    else:
        st.success("✅ Handled by AI")

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.last_metadata = {"persona": "Not Detected", "confidence": 0.0, "escalated": False, "reason": "None"}
        st.rerun()

# --- Orchestrator Initialization ---
@st.cache_resource
def get_orchestrator(api_key, model):
    if api_key:
        return SupportOrchestrator(api_key=api_key, model_name=model)
    return None

orchestrator = get_orchestrator(openrouter_key, model_choice)

# --- Main Chat UI ---
st.title("💬 AdsSparkX Customer Support")
st.caption("Integrated Persona-Adaptive RAG Agent (LangChain + OpenRouter)")

# Initialize session messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input loop
if prompt := st.chat_input("How can I help you today?"):
    if not openrouter_key:
        st.error("Missing OpenRouter API Key. Please provide it in the sidebar.")
    else:
        # Add user message to UI
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Process Request
        with st.chat_message("assistant"):
            try:
                with st.spinner("Analyzing..."):
                    data = orchestrator.process_request(prompt)
                    
                    ai_response = data["response"]
                    st.markdown(ai_response)
                    
                    # Update session state
                    st.session_state.messages.append({"role": "assistant", "content": ai_response})
                    st.session_state.last_metadata = {
                        "persona": data["persona_info"]["persona"],
                        "confidence": data["response_info"]["confidence"],
                        "escalated": data["escalation"]["escalation"],
                        "reason": data["escalation"]["reason"]
                    }
                    st.rerun() 
                    
            except Exception as e:
                st.error(f"Processing Error: {str(e)}")
                st.info("Check your API key and ensure you have built the KB index.")
