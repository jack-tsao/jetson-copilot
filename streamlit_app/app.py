import ollama
import openai
import streamlit as st

from llama_index.core import VectorStoreIndex, Settings, SimpleDirectoryReader
from llama_index.core import load_index_from_storage, StorageContext
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.llms.ollama import Ollama
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from PIL import Image
import time

import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

import utils.func 
import utils.constants as const

# App title
st.set_page_config(page_title="Advantech Copilot", menu_items=None)

AVATAR_AI   = Image.open('./images/jetson-soc.png')
AVATAR_USER = Image.open('./images/user-purple.png')

def find_saved_indexes():
    return utils.func.list_directories(const.INDEX_ROOT_PATH)

def load_index(index_name):
    Settings.embed_model = OllamaEmbedding("llama3:latest") ##TODO
    logging.info(f"Loading index with embedding model: llama3:latest")

    dir = f"{const.INDEX_ROOT_PATH}/{index_name}"
    storage_context = StorageContext.from_defaults(persist_dir=dir)
    index = load_index_from_storage(storage_context)
    return index


models = [model["name"] for model in ollama.list()["models"]]

if 'llama3:latest' not in models:
    with st.spinner('Downloaing llama3 model ...'):
        ollama.pull('llama3')
        logging.info(" ### Downloaing llama3 completed.")

if 'mxbai-embed-large:latest' not in models:
    with st.spinner('Downloaing mxbai-embed-large model ...'):
        ollama.pull('mxbai-embed-large')
        logging.info(" ### Downloaing mxbai-embed-large completed.")

old_index_name = ''
# Side bar
with st.sidebar:        
    # # Add css to make text smaller
    # st.markdown(
    #     """<style>textarea { font-size: 0.8rem !important; } </style>""",
    #     unsafe_allow_html=True,
    # )
    st.title(":airplane: Advantech Copilot")
    st.subheader('Your local AI assistant on Advantech Documents', divider='rainbow')

    models = [model["name"] for model in ollama.list()["models"]]
    col3, col4 = st.columns([5,1])
    with col3:
        st.session_state["model"] = st.selectbox("Choose your LLM", models, index=models.index("llama3:latest"))
        logging.info(f"> st.session_state[\"model\"] = {st.session_state.model}")
    with col4:
        st.markdown('')
        # st.button('➕', key='btn_add_llm')
    st.page_link("pages/download_model.py", label=" Download a new LLM", icon="➕")
    Settings.llm = Ollama(model=st.session_state["model"], request_timeout=300.0)

    use_index = st.toggle("Use RAG", value=False)
    if use_index:
        # col1, col2 = st.columns([5,1], vertical_alignment="bottom") ### https://github.com/streamlit/streamlit/issues/3052
        col1, col2 = st.columns([5,1])
        saved_index_list = find_saved_indexes()
        with col1:
            index = next((i for i, item in enumerate(saved_index_list) if item.startswith('_')), None)
            index_name = st.selectbox("Index", saved_index_list, index)
            logging.info(f"> index_name = {index_name}")
        with col2:
            st.markdown('')
            # st.link_button('➕', url='pages/build_index.py')
        if old_index_name != index_name:
            old_index_name = index_name
            logging.info(f"> old_index_name = {old_index_name}")
            if index_name != None:
                with st.spinner('Loading Index...'):
                    st.session_state.index = load_index(index_name)
                    logging.info(f" ### Loading Index '{index_name}' completed.")
        st.page_link("pages/build_index.py", label=" Build a new index", icon="➕")

        if index_name != None:
            context_prompt = st.text_area("System prompt with context", 
"""ROLE:
You are a precise knowledge assistant that provides accurate answers based solely on the provided documents.

DOCUMENT:
{context_str}

QUESTION:
{query_str}

INSTRUCTIONS:
1. First, identify the specific sections of the DOCUMENT that are most relevant to the QUESTION
2. Quote these relevant sections verbatim and indicate why they are applicable
3. Then answer the QUESTION using only information from these sections
4. Format your response in a clear, structured way
5. If the DOCUMENT doesn't contain sufficient information to answer the QUESTION, respond with INSUFFICIENT INFORMATION and suggest what additional details would be needed""", height=340)
            logging.info(f"> context_prompt = {context_prompt}")

            # init models
            st.session_state.chat_engine = st.session_state.index.as_chat_engine(
                chat_mode="context", 
                streaming=True,
                memory=ChatMemoryBuffer.from_defaults(token_limit=4096),
                llm=Settings.llm,
                context_prompt=(context_prompt),
                verbose=True)

# initialize history
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "👋 Welcome to Advantech Copilot! I can answer questions and help with your AI projects using the knowledge in your vectorized documents.\n\n"
        "**For best results with RAG (when enabled):**\n"
        "- Be specific and detailed in your questions\n"
        "- Break down complex questions into smaller, focused parts\n"
        "- Include specific keywords from your documents\n"
        "- Mention technical specifications when relevant\n\n"
        "What would you like to know today?", 
        "avatar": AVATAR_AI}]
    

def model_res_generator(prompt=""):
    if use_index:
        logging.info(f">>> RAG enabled:")

        # Add this logging before stream_chat to see the actual query
        logging.info(f">>> Query: {prompt}")
        
        # Get the retrieved nodes before streaming to see what chunks are being used
        retriever = st.session_state.index.as_retriever(
            similarity_top_k=5  # Adjust this number based on your needs
        )
        retrieved_nodes = retriever.retrieve(prompt)
        
        # Log the retrieved chunks
        logging.info(f">>> Retrieved {len(retrieved_nodes)} chunks for context:")
        for i, node in enumerate(retrieved_nodes):
            logging.info(f">>> Chunk {i+1}:")
            logging.info(f">>> Score: {node.score if hasattr(node, 'score') else 'N/A'}")
            logging.info(f">>> Text: {node.text}") 
        
        response_stream = st.session_state.chat_engine.stream_chat(prompt)
        for chunk in response_stream.response_gen:
            yield chunk
    else:
        logging.info(f">>> Just LLM (no RAG):")
        messages_only_role_and_content = [{"role": message["role"], "content": message["content"]} for message in st.session_state.messages]

        stream = ollama.chat(
            model=st.session_state["model"],
            messages=messages_only_role_and_content,
            stream=True,
        )
        for chunk in stream:
            yield chunk["message"]["content"]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=message["avatar"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Enter prompt here.."):
    # add latest message to history in format {role, content}
    st.session_state.messages.append({"role": "user", "content": prompt, "avatar": AVATAR_USER})

    with st.chat_message("user", avatar=AVATAR_USER):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=AVATAR_AI):
        with st.spinner("Thinking..."):
            time.sleep(1)
            message = st.write_stream(model_res_generator(prompt))
            st.session_state.messages.append({"role": "assistant", "content": message, "avatar": AVATAR_AI})
