"""Interface main function."""

import requests
import streamlit as st

# Apply custom CSS for title font size
st.markdown(
    """
    <style>
    h1 {
        font-size: 40px;
    }
    .stButton {
        text-align: right;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Display the main title
st.title("Kubernetes Documentation Assistant")


# Initialize the session state for messages if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display the chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

# Handle new user input
if prompt := st.chat_input("How may I help you?"):
    # Store the user input in the session state and display it in the chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get response from the API and display it
    with st.chat_message("assistant"):
        response = requests.post("http://api:8000/chat", json={"text": prompt})
        response_text = response.text.strip('"')
        response_text = response_text.replace("\\n", "\n")
        response_text = response_text.replace("**질문**", "**질문**\n")
        response_text = response_text.replace("**답변**", "**답변**\n")
        response_text = response_text.replace("**관련 문서**", "**관련 문서**\n")

        st.markdown(response_text, unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": response_text})

# Clear chat history when the Clear button is pressed
if st.button("Clear"):
    st.session_state.messages = []
    requests.post("http://api:8000/clear")
    st.rerun()
