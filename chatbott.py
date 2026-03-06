import os 
import json 
import time 
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory



# load env
load_dotenv()
ENV_GROQ_API_KEY = os.getenv("GROQ_API_KEY","").strip()


# Streamlit page config
st.set_page_config(
    page_title="Groq Chatbot (With Memory)",
    page_icon="🤖",
    layout="centered"
)

# ─────────────────────────────────────────────
# PURPLE THEME INJECTION
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* Root vars */
:root {
    --purple-deep:   #1a0a2e;
    --purple-dark:   #16213e;
    --purple-mid:    #7b2fff;
    --purple-light:  #a855f7;
    --purple-glow:   #c084fc;
    --purple-soft:   #2d1b69;
    --accent-pink:   #f472b6;
    --accent-cyan:   #22d3ee;
    --text-primary:  #f0e6ff;
    --text-muted:    #a78bca;
    --glass:         rgba(123, 47, 255, 0.08);
    --glass-border:  rgba(168, 85, 247, 0.25);
}

/* Global */
html, body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0d0118 0%, #1a0a2e 40%, #0f172a 100%) !important;
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text-primary) !important;
}

[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    top: -40%;
    left: -20%;
    width: 80%;
    height: 80%;
    background: radial-gradient(ellipse, rgba(123,47,255,0.18) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
}

[data-testid="stAppViewContainer"]::after {
    content: '';
    position: fixed;
    bottom: -30%;
    right: -10%;
    width: 60%;
    height: 60%;
    background: radial-gradient(ellipse, rgba(244,114,182,0.10) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
}

/* Main content area */
[data-testid="stMain"] {
    background: transparent !important;
}

/* Title */
h1 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
    font-size: 2.2rem !important;
    background: linear-gradient(90deg, #c084fc, #f472b6, #22d3ee) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    letter-spacing: -0.5px !important;
}

/* Caption */
[data-testid="stCaptionContainer"] p {
    color: var(--text-muted) !important;
    font-size: 0.85rem !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #130826 0%, #1a0a2e 60%, #0f172a 100%) !important;
    border-right: 1px solid var(--glass-border) !important;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] label {
    color: var(--text-primary) !important;
    font-family: 'Syne', sans-serif !important;
}

[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stTextArea label,
[data-testid="stSidebar"] .stTextInput label,
[data-testid="stSidebar"] .stCheckbox label {
    color: var(--purple-glow) !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.3px !important;
}

/* Sidebar subheader */
[data-testid="stSidebar"] h3 {
    font-size: 0.78rem !important;
    text-transform: uppercase !important;
    letter-spacing: 1.5px !important;
    color: var(--text-muted) !important;
    margin-top: 1rem !important;
    margin-bottom: 0.3rem !important;
}

/* Select boxes */
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: var(--glass) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
}

/* Text inputs */
[data-testid="stSidebar"] .stTextInput input,
[data-testid="stSidebar"] .stTextArea textarea {
    background: var(--glass) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stSidebar"] .stTextInput input:focus,
[data-testid="stSidebar"] .stTextArea textarea:focus {
    border-color: var(--purple-mid) !important;
    box-shadow: 0 0 0 2px rgba(123,47,255,0.3) !important;
}

/* Sliders */
[data-testid="stSidebar"] .stSlider [data-testid="stSlider"] > div > div > div {
    background: linear-gradient(90deg, var(--purple-mid), var(--accent-pink)) !important;
}
[data-testid="stSidebar"] .stSlider [role="slider"] {
    background: var(--purple-glow) !important;
    border: 2px solid white !important;
    box-shadow: 0 0 8px var(--purple-mid) !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--purple-mid), var(--purple-light)) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.3px !important;
    padding: 0.45rem 1rem !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 2px 12px rgba(123,47,255,0.35) !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(123,47,255,0.55) !important;
    background: linear-gradient(135deg, #9333ea, var(--accent-pink)) !important;
}

/* Download buttons */
.stDownloadButton > button {
    background: linear-gradient(135deg, rgba(123,47,255,0.2), rgba(244,114,182,0.15)) !important;
    color: var(--purple-glow) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.80rem !important;
    padding: 0.4rem 0.8rem !important;
    transition: all 0.25s ease !important;
    width: 100% !important;
}

.stDownloadButton > button:hover {
    background: linear-gradient(135deg, rgba(123,47,255,0.4), rgba(244,114,182,0.25)) !important;
    border-color: var(--purple-light) !important;
    color: white !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 14px rgba(123,47,255,0.35) !important;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    background: var(--glass) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 16px !important;
    padding: 0.8rem 1rem !important;
    margin-bottom: 0.5rem !important;
    backdrop-filter: blur(8px) !important;
}

/* User message accent */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    border-left: 3px solid var(--accent-pink) !important;
    background: linear-gradient(135deg, rgba(244,114,182,0.07), rgba(123,47,255,0.07)) !important;
}

/* Assistant message accent */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    border-left: 3px solid var(--purple-mid) !important;
    background: linear-gradient(135deg, rgba(123,47,255,0.07), rgba(34,211,238,0.05)) !important;
}

/* Chat input */
[data-testid="stChatInput"] textarea {
    background: rgba(45, 27, 105, 0.5) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 14px !important;
    color: var(--text-primary) !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stChatInput"] textarea:focus {
    border-color: var(--purple-mid) !important;
    box-shadow: 0 0 0 2px rgba(123,47,255,0.3), 0 0 20px rgba(123,47,255,0.15) !important;
}

[data-testid="stChatInput"] button {
    background: linear-gradient(135deg, var(--purple-mid), var(--accent-pink)) !important;
    border-radius: 10px !important;
}

/* Divider */
hr {
    border-color: var(--glass-border) !important;
    opacity: 0.5 !important;
}

/* Checkbox */
[data-testid="stSidebar"] .stCheckbox span {
    color: var(--text-primary) !important;
}

/* Error */
[data-testid="stAlert"] {
    background: rgba(239,68,68,0.15) !important;
    border: 1px solid rgba(239,68,68,0.35) !important;
    border-radius: 12px !important;
    color: #fca5a5 !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: linear-gradient(var(--purple-mid), var(--accent-pink));
    border-radius: 99px;
}
</style>
""", unsafe_allow_html=True)


st.title("🤖 Conversational AI Chatbot")
st.caption("Built with Streamlit + Langchain + Groq Cloud API")


# -----------------------------
# SIDEBAR CONTROL
# -----------------------------
with st.sidebar:
    st.header("⚙️ Controls")

    api_key_input = st.text_input(
        "Groq API Key (optional)",
        type="password"
    )
    GROQ_API_KEY = api_key_input.strip() if api_key_input.strip() else ENV_GROQ_API_KEY


    # ---------------------------------------------------------
    # ADDED 2 NEW MODELS 
    # ---------------------------------------------------------
    model_name = st.selectbox(
        "Choose Model",
        [
            "openai/gpt-oss-120b",
            "llama-3.3-70b-versatile",
            "qwen/qwen3-32b",
            "mixtral-8x7b-instruct",             
            "llama-3.1-8b-instant"              
        ],
        index=1
    )

    temperature = st.slider(
        "Temperature (Creativity)",
        min_value=0.0,
        max_value=1.0,
        value=0.4,
        step=0.1
    )

    max_tokens = st.slider(
        "Max Tokens (Reply length)",
        min_value=64,
        max_value=1024,
        value=256,
        step=64
    )

    # ---------------------------------------------------------
    # ADDED TONE PRESET DROPDOWN
    # ---------------------------------------------------------
    tone_preset = st.selectbox(
        "Tone Preset",
        ["Friendly", "Strict", "Teacher"]
    )

    # Tone mapping
    TONE_MAP = {
        "Friendly": "Respond in a warm, helpful, friendly manner.",
        "Strict": "Respond in a serious, concise, strict tone.",
        "Teacher": "Respond like a knowledgeable teacher, explaining concepts clearly.",
    }

    system_prompt = st.text_area(
        "System Prompt (Rules for the bot)",
        value="You are helpful AI Assistant. Be Clear, correct and concise.",
        height=140
    )

    # -----------------------------------------
    # RESET SYSTEM PROMPT BUTTON 
    # -----------------------------------------
    if st.button("Reset System Prompt"):
        st.session_state["system_prompt"] = "You are helpful AI Assistant. Be Clear, correct and concise."
        st.rerun()

    typing_effect = st.checkbox("Enable typing effect", value=True)

    st.divider()

    if st.button(" 🧹 Clear Chat "):
        st.session_state.pop("history_store", None)
        st.session_state.pop("download_cache", None)
        st.rerun()

    # ---------------------------
    # DOWNLOAD CHAT SECTION (moved to sidebar)
    # ---------------------------
    st.divider()
    st.subheader("⬇️ Download Chat")

    # Build export data for sidebar downloads
    if "history_store" in st.session_state:
        sidebar_export = []
        _hist = st.session_state.history_store.get("default_session")
        if _hist:
            for m in _hist.messages:
                role = getattr(m, "type", "")
                if role == "human":
                    sidebar_export.append({"role": "user", "text": m.content})
                else:
                    sidebar_export.append({"role": "assistant", "text": m.content})
    else:
        sidebar_export = []

    sidebar_txt = "\n\n".join([f"{m['role'].upper()}: {m['text']}" for m in sidebar_export])

    st.download_button(
        label="📄 chat_history.json",
        data=json.dumps(sidebar_export, ensure_ascii=False, indent=2),
        file_name="chat_history.json",
        mime="application/json",
    )

    st.download_button(
        label="📝 chat_history.txt",
        data=sidebar_txt,
        file_name="chat_history.txt",
        mime="text/plain",
    )


# Guard
if not GROQ_API_KEY:
    st.error("🔑 Groq API Key is missing. Add it in your .env or paste it in the sidebar")
    st.stop()


# chat history store
if "history_store" not in st.session_state:
    st.session_state.history_store = {}

SESSION_ID = "default_session"

def get_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in st.session_state.history_store:
        st.session_state.history_store[session_id] = InMemoryChatMessageHistory()
    return st.session_state.history_store[session_id]


# ---------------------------------------------------------
# TONE PRESET IS  MERGED INTO SYSTEM PROMPT
# ---------------------------------------------------------
full_system_prompt = f"{system_prompt}\n\nTone Instruction: {TONE_MAP[tone_preset]}"


# Build LLM
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model=model_name,
    temperature=temperature,
    max_tokens=max_tokens
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "{system_prompt}"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

chain = prompt | llm | StrOutputParser() 

chat_with_history = RunnableWithMessageHistory(
    chain,
    get_history,
    input_messages_key="input",
    history_messages_key="history"
)


# ------------------------
# Render old messages
# ------------------------
history_obj = get_history(SESSION_ID)

for msg in history_obj.messages:
    role = getattr(msg, "type", "")
    if role == "human":
        st.chat_message("user").write(msg.content)
    else:
        st.chat_message("assistant").write(msg.content)


# ------------------------
# User input + Model response
# ------------------------
user_input = st.chat_input("Type your message....")

if user_input:
    st.chat_message("user").write(user_input)

    with st.chat_message("assistant"):
        placeholder = st.empty()

        try:
            response_text = chat_with_history.invoke(
                {"input": user_input, "system_prompt": full_system_prompt},
                config={"configurable": {"session_id": SESSION_ID}},
            )
        except Exception as e:
            st.error(f"Model Error: {e}")
            response_text = ""

        if typing_effect and response_text:
            typed = ""
            for ch in response_text:
                typed += ch
                placeholder.markdown(typed)
                time.sleep(0.005)
        else:
            placeholder.write(response_text)