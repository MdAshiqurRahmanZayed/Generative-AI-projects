from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
import streamlit as st
import re
import json

st.title("Langchain-DeepSeek-R1 app")

CHAT_HISTORY_FILE = "chat_history.json"


def load_chat_history():
    try:
        with open(CHAT_HISTORY_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def save_chat_history(chat_history):
    with open(CHAT_HISTORY_FILE, "w") as f:
        json.dump(chat_history, f)


template = """Question: {question}

Answer: Let's think step by step."""

prompt = ChatPromptTemplate.from_template(template)

# Initialize the model
model = OllamaLLM(model="deepseek-r1:1.5b", base_url="http://localhost:11434")
chain = prompt | model

st.session_state.chat_history = load_chat_history()

question = st.chat_input("Enter your question here")
if question:
    st.session_state.chat_history.append({"role": "user", "content": question})
    response = chain.invoke({"question": question})

    # Replace any instance of <think> ... </think> with a thinking emoji
    response = re.sub(r"<think>.*?</think>", "🧠", response, flags=re.DOTALL)

    st.session_state.chat_history.append({"role": "assistant", "content": response})
    save_chat_history(st.session_state.chat_history)

# Display chat history
for chat in st.session_state.chat_history:
    if chat["role"] == "user":
        st.write(f"**You:** {chat['content']}")
    else:
        st.write(f"**AI:** {chat['content']}")
