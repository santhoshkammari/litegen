import streamlit as st
import requests
from typing import Generator

from typing_extensions import Literal

MODELS_TYPE = Literal[
    'meta-llama/Meta-Llama-3.1-70B-Instruct',
    'CohereForAI/c4ai-command-r-plus-08-2024',
    'Qwen/Qwen2.5-72B-Instruct',
    'nvidia/Llama-3.1-Nemotron-70B-Instruct-HF',
    'meta-llama/Llama-3.2-11B-Vision-Instruct',
    'NousResearch/Hermes-3-Llama-3.1-8B',
    'mistralai/Mistral-Nemo-Instruct-2407',
    'microsoft/Phi-3.5-mini-instruct'
]


def get_api_response(prompt, model=None, stream=True, conversation=False, url="http://0.0.0.0:11435/v1/generate",
                     web_search=False) -> Generator:
    data = {
        "prompt": prompt,
        "model": model,
        "stream": stream,
        "conversation": conversation,
        "websearch": web_search
    }
    response = requests.post(url, json=data, stream=stream)
    for chunk in response.iter_content(decode_unicode=True):
        if chunk:
            yield chunk


def streamlit_app(model:MODELS_TYPE='nvidia/Llama-3.1-Nemotron-70B-Instruct-HF'):
    st.title("Chat with LLMs")

    # Add a separator
    st.markdown("---")

    # Chat messages container
    chat_container = st.container()

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if 'conversation_mode' not in st.session_state:
        st.session_state.conversation_mode = True
    if 'web_search' not in st.session_state:
        st.session_state.web_search = False
    if 'model' not in st.session_state:
        st.session_state.model = model  # Default model

    # Display chat messages
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("What is your message?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = message_placeholder.write_stream(get_api_response(
                    prompt,
                    model=st.session_state.model,
                    conversation=st.session_state.conversation_mode,
                    web_search=st.session_state.web_search))
        st.session_state.messages.append({"role": "assistant", "content": full_response})

    # Create a container for controls (model selector, toggles, and clear button)
    with st.container():
        col2, col3, col4 = st.columns(3)


        with col2:
            st.session_state.conversation_mode = st.toggle("Conversation Mode", value=True)
        with col3:
            st.session_state.web_search = st.toggle("Enable Web Search", value=False)
        with col4:
            if st.button("Clear Chat History", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

    with st.container():
        st.session_state.model = st.selectbox(
            "Select Model",
            options=list(MODELS_TYPE.__args__),
            index=list(MODELS_TYPE.__args__).index(st.session_state.model),
            label_visibility='hidden'
        )


if __name__ == "__main__":
    streamlit_app()