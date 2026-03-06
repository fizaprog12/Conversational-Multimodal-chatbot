# Conversational-Multimodal-chatbot
- A conversational AI chatbot built with Streamlit, LangChain, and Groq Cloud API — featuring persistent memory, multiple LLM models, tone presets, and chat export.

✨ Features:

- 🧠 Persistent Chat Memory — Remembers the full conversation using InMemoryChatMessageHistory
- 🤖 Multi-Model Support — Switch between 5 LLMs from the sidebar
- 🎨 Tone Presets — Choose between Friendly, Strict, or Teacher tone
- ✍️ Custom System Prompt — Define the bot's behavior with your own rules
- ⚡ Typing Effect — Simulated character-by-character streaming response
- 📥 Chat Export — Download full conversation as .json or .txt
- 🧹 Clear Chat — Reset conversation history in one click
- 🔑 Flexible API Key — Provide via .env file or directly in the sidebar

🛠️ Tech Stack:
- Layer              |      Technology
- >Frontend          |      Streamlit
- >LLMOrchestration  |      LangChain
- >LLMProviderGroq   |      Cloud API
- >MemoryLangChain   |      InMemoryChatMessageHistory
- >Environment       |      Python-dotenv
