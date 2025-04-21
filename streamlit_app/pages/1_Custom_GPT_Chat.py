import streamlit as st
import openai
from PIL import Image
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(page_title="Custom GPT Chat", layout="wide")

# Add return to main page button
if st.sidebar.button("Return to Main Page", type="secondary"):
    st.switch_page("app.py")

# Load images
AVATAR_AI = Image.open('./images/jetson-soc.png')
AVATAR_USER = Image.open('./images/user-purple.png')

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Title and description
st.title("Custom GPT Chat")
st.markdown("Chat with your custom GPT assistant. Ask questions and get personalized responses!")

# Add configuration in sidebar
with st.sidebar:
    st.subheader("Configuration")
    model = st.selectbox(
        "Select Model",
        ["gpt-4", "gpt-3.5-turbo"],
        help="Select the base model for your custom GPT"
    )
    temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.1,
                          help="Higher values make the output more random, lower values make it more deterministic")

# Chat interface
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=AVATAR_AI if message["role"] == "assistant" else AVATAR_USER):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What would you like to ask?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=AVATAR_USER):
        st.markdown(prompt)

    # Get GPT response
    with st.chat_message("assistant", avatar=AVATAR_AI):
        with st.spinner("Thinking..."):
            try:
                # Initialize OpenAI client
                client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                
                # Get response from GPT
                response = client.chat.completions.create(
                    model=model,  # Use the selected base model
                    messages=[
                        {"role": "system", "content": "You are a helpful AI assistant."},
                        *[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=1000
                )
                
                # Display the response
                response_content = response.choices[0].message.content
                st.markdown(response_content)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response_content})
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.info("""
                    Please check:
                    1. Your API key is valid and has sufficient credits
                    2. You have access to the selected model
                    3. Your network connection is working
                """)

# Add a clear chat button
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun() 