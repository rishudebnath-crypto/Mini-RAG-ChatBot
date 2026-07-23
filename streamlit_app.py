import streamlit as st
import requests

st.set_page_config(
    page_title="Mini RAG Chatbot"
)

st.title("Welcome! This platform allows you to converse with the Groq LLM to ask RAG based questions on the PDF document about adiabatic computing. Start a conversation, save the chats, and delete them at your will")
st.write("Ask questions about your uploaded document.")





# -----------------------------
# Load chat history once
# -----------------------------
if "messages" not in st.session_state:

    history = requests.get(
        "http://127.0.0.1:8000/history"
    ).json()

    st.session_state.messages = history

# -----------------------------
# Display previous conversation
# -----------------------------
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.write(message["content"])

# -----------------------------
# Chat input
# -----------------------------
question = st.chat_input("Ask something...")

if question:

    # Display user message
    with st.chat_message("user"):
        st.write(question)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    # Send request to FastAPI
    response = requests.post(
        "http://127.0.0.1:8000/chat",
        json={
            "question": question
        }
    )

    answer = response.json()["response"]

    # Display assistant response
    with st.chat_message("assistant"):
        st.write(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:

    st.header("Options")

    # Download PDF
    with open("Adiabatic_computing.pdf", "rb") as pdf:

        st.download_button(
            label="📄 Download Original PDF",
            data=pdf,
            file_name="quantum_paper.pdf",
            mime="application/pdf"
        )

    st.divider()

    # Clear chat button
    if st.button("🗑️ Clear Chat"):

        requests.delete(
            "http://127.0.0.1:8000/history"
        )

        st.session_state.messages = []

        st.rerun()