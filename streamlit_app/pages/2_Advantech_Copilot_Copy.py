import ollama
import openai
import streamlit as st
from langdetect import detect

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

# Set page config first
st.set_page_config(page_title="Advantech Copilot (COPY)", menu_items=None)

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
import utils.func 
import utils.constants as const
AVATAR_SYS  = Image.open('./images/Cog-box-01.png')
AVATAR_AI   = Image.open('./images/jetson-soc.png')
AVATAR_USER = Image.open('./images/user-purple.png')
ADV_LOGO    = Image.open('./images/advlogo2.png')

# Add navigation buttons
col1, col2 = st.sidebar.columns([1, 1])
with col1:
    if st.button("Return to Main", type="secondary"):
        st.switch_page("app.py")
with col2:
    if st.button("Chat with Custom GPT", type="primary"):
        st.switch_page("pages/1_Custom_GPT_Chat.py")

def find_saved_indexes():
    return utils.func.list_directories(const.INDEX_ROOT_PATH)

def load_index(index_name):
    Settings.embed_model = OllamaEmbedding("mxbai-embed-large:latest")
    logging.info("Loading index with embedding model: mxbai-embed-large:latest")
    dir = f"{const.INDEX_ROOT_PATH}/{index_name}"
    storage_context = StorageContext.from_defaults(persist_dir=dir)
    index = load_index_from_storage(storage_context)
    return index

# Add model-specific configurations
MODEL_CONFIGS = {
    "llama3:latest": {
        "system_template": """You are a multilingual AI assistant. You MUST respond in {lang} language using proper {lang} characters.
For Japanese: Use ひらがな, カタカナ, and 漢字 appropriately
For Chinese: Use 简体中文 characters
For English: Use standard English characters

DO NOT mix languages or use English characters when responding in Japanese or Chinese.
DO NOT use romaji or pinyin when responding in Japanese or Chinese.

{base_instructions}"""
    },
    "glm-4:latest": {
        "system_template": """You are a multilingual AI assistant. You MUST respond in {lang} language using proper {lang} characters.
For Japanese: Use ひらがな, カタカナ, and 漢字 appropriately
For Chinese: Use 简体中文 characters
For English: Use standard English characters

DO NOT mix languages or use English characters when responding in Japanese or Chinese.
DO NOT use romaji or pinyin when responding in Japanese or Chinese.

{base_instructions}"""
    }
}

models = [model["name"] for model in ollama.list()["models"]]

if 'llama3:latest' not in models:
    with st.spinner('Downloaing llama3 model ...'):
        ollama.pull('llama3')
        logging.info("### Downloaing llama3 completed.")

if 'mxbai-embed-large:latest' not in models:
    with st.spinner('Downloaing mxbai-embed-large model ...'):
        ollama.pull('mxbai-embed-large')
        logging.info("### Downloaing mxbai-embed-large completed.")

old_index_name = ''

# ---------------------------
# Sidebar remains largely the same (except we remove the system prompt here)
with st.sidebar:        
    # Replace text title with logo
    st.image(ADV_LOGO, width=200)
    st.subheader('Your local AI assistant on Advantech Documents (Copy)', divider='rainbow')

    models = [model["name"] for model in ollama.list()["models"]]
    col3, col4 = st.columns([5, 1])
    with col3:
        st.session_state["model"] = st.selectbox("Choose your LLM", models, index=models.index("llama3:latest"))
    with col4:
        st.markdown('')
    st.page_link("pages/download_model.py", label=" Download a new LLM", icon="➕")
    
    # Configure LLM with appropriate settings
    try:
        Settings.llm = Ollama(
            model=st.session_state["model"],
            request_timeout=300.0,
            temperature=0.7,
            context_window=4096
        )
    except Exception as e:
        st.error(f"Error configuring model {st.session_state['model']}: {str(e)}")
        st.session_state["model"] = "llama3:latest"  # Fallback to llama3
        Settings.llm = Ollama(model="llama3:latest", request_timeout=300.0)

    use_index = st.toggle("Use RAG", value=False)
    if use_index:
        col1, col2 = st.columns([5, 1])
        saved_index_list = find_saved_indexes()
        with col1:
            index = next((i for i, item in enumerate(saved_index_list) if item.startswith('_')), None)
            index_name = st.selectbox("Index", saved_index_list, index)
        with col2:
            st.markdown('')
            
        # Only load index if it's different from the current one
        if "current_index" not in st.session_state or st.session_state.current_index != index_name:
            if index_name is not None:
                with st.spinner('Loading Index...'):
                    st.session_state.index = load_index(index_name)
                    st.session_state.current_index = index_name
                    # Initialize chat engine with current language
                    lang = st.session_state.get("prompt_language", "English")
                    base_instructions = f"""ROLE:
You are a precise knowledge assistant that provides accurate answers based solely on the provided documents.
You MUST respond in {lang} language ONLY. Use proper {lang} characters and formatting.
DO NOT use English characters or formatting in your response.

DOCUMENT:
{{context_str}}

QUESTION:
{{query_str}}

INSTRUCTIONS:
1. First, identify the specific sections of the DOCUMENT that are most relevant to the QUESTION
2. Quote these relevant sections verbatim and indicate why they are applicable
3. Then answer the QUESTION using only information from these sections
4. Format your response in a clear, structured way using proper {lang} characters
5. If the DOCUMENT doesn't contain sufficient information to answer the QUESTION, respond with INSUFFICIENT INFORMATION and suggest what additional details would be needed"""
                    
                    st.session_state.chat_engine = st.session_state.index.as_chat_engine(
                        chat_mode="context", 
                        streaming=True,
                        memory=ChatMemoryBuffer.from_defaults(token_limit=4096),
                        llm=Settings.llm,
                        context_prompt=base_instructions,
                        verbose=True)
        
        st.page_link("pages/build_index.py", label=" Build a new index", icon="➕")
        st.markdown("---")
        
        # Move system prompt to sidebar with model-specific template
        lang = st.session_state.get("prompt_language", "English")
        base_instructions = f"""ROLE:
You are a precise knowledge assistant that provides accurate answers based solely on the provided documents.
You MUST respond in {lang} language ONLY. Use proper {lang} characters and formatting.
DO NOT use English characters or formatting in your response.

DOCUMENT:
{{context_str}}

QUESTION:
{{query_str}}

INSTRUCTIONS:
1. First, identify the specific sections of the DOCUMENT that are most relevant to the QUESTION
2. Quote these relevant sections verbatim and indicate why they are applicable
3. Then answer the QUESTION using only information from these sections
4. Format your response in a clear, structured way using proper {lang} characters
5. If the DOCUMENT doesn't contain sufficient information to answer the QUESTION, respond with INSUFFICIENT INFORMATION and suggest what additional details would be needed"""

        # Get model-specific template
        model_config = MODEL_CONFIGS.get(st.session_state["model"], MODEL_CONFIGS["llama3:latest"])
        system_prompt = model_config["system_template"].format(
            lang=lang,
            base_instructions=base_instructions
        )
        
        context_prompt = st.text_area("System prompt with context", value=system_prompt, height=340)
        if hasattr(st.session_state, 'chat_engine'):
            st.session_state.chat_engine.context_prompt = context_prompt

# Initialize chat history
if "messages" not in st.session_state.keys():
    initial_message = (
        "Welcome to Advantech Copilot (Copy)!\n\n"
        "**RAG status: Disabled.**\n\n"
        "Currently, our system is set to deliver fast general responses. For detailed, document-specific answers, please enable RAG.\n"
        "**Tips:**\n"
        "- Enable RAG in settings to access detailed retrieval.\n"
        "- Use general queries for quick responses if RAG is disabled.\n"
        "- Consider turning on RAG when asking specific questions about documents.\n"
     )
    st.session_state.messages = [{
        "role": "assistant", 
        "content": initial_message, 
        "avatar": AVATAR_AI
    }]
# initialize previous state of RAG
if "prev_use_index" not in st.session_state:
    st.session_state.prev_use_index = use_index
# Check if RAG status has changed
if st.session_state.prev_use_index != use_index:
    st.session_state.prev_use_index = use_index
    rag_status_message = (
        "**RAG status: :green[Enabled].**\n\n"
        "Please select the appropriate index from the sidebar."
         if use_index else 
         "**RAG status: :red[Disabled].**\n\n"
         "For detailed, document-specific answers, please enable RAG."
    )
    st.session_state.messages.append({
        "role": "system",
        "content": rag_status_message,
        "avatar": AVATAR_SYS
    })

# Display chat messages (above the bottom fixed container)
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=message["avatar"]):
        st.markdown(message["content"])

# ---------------------------
# Fixed bottom container for language selection and chat input
st.markdown("""
    <style>
    .fixed-bottom-container {
        position: fixed;
        left: 0;
        right: 0;
        bottom: 0;
        background-color: white;
        padding: 10px;
        border-top: 1px solid #ddd;
        z-index: 1000;
    }
    .main .block-container {
        padding-bottom: 180px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="fixed-bottom-container">', unsafe_allow_html=True)

# Create columns for language selection and chat input
lang_col, chat_col = st.columns([1, 3])

with lang_col:
    # Use selectbox instead of radio for more reliable state management
    selected_lang = st.selectbox(
        "Language",
        ["中文", "English", "日本語"],
        key="lang_select",
        label_visibility="collapsed"
    )

# Map the displayed values to internal language strings
lang_map = {"中文": "Chinese", "English": "English", "日本語": "Japanese"}
current_lang = lang_map[selected_lang]

# Update system prompt if language changed
if use_index and (index_name is not None):
    base_instructions = f"""ROLE:
You are a precise knowledge assistant that provides accurate answers based solely on the provided documents.
You MUST respond in {current_lang} language ONLY. Use proper {current_lang} characters and formatting.
DO NOT use English characters or formatting in your response.

DOCUMENT:
{{context_str}}

QUESTION:
{{query_str}}

INSTRUCTIONS:
1. First, identify the specific sections of the DOCUMENT that are most relevant to the QUESTION
2. Quote these relevant sections verbatim and indicate why they are applicable
3. Then answer the QUESTION using only information from these sections
4. Format your response in a clear, structured way using proper {current_lang} characters
5. If the DOCUMENT doesn't contain sufficient information to answer the QUESTION, respond with INSUFFICIENT INFORMATION and suggest what additional details would be needed"""

    # Update the chat engine with new language settings
    if hasattr(st.session_state, 'chat_engine'):
        st.session_state.chat_engine.context_prompt = base_instructions

with chat_col:
    # Chat input field
    prompt = st.chat_input("Enter prompt here...")

st.markdown('</div>', unsafe_allow_html=True)  # Close fixed bottom container
# ---------------------------

def detect_language(text: str) -> str:
    """Detect the language of the input text."""
    try:
        return detect(text)
    except:
        return "en"

def get_system_prompt(lang: str) -> str:
    """Get the system prompt in the detected language."""
    prompts = {
        "en": """You are a precise knowledge assistant that provides accurate answers based solely on the provided documents.
        Respond in English using standard English characters.""",
        "ja": """あなたは提供された文書に基づいて正確な回答を提供する知識アシスタントです。
        ひらがな、カタカナ、漢字を使用して日本語で回答してください。""",
        "zh": """你是一个基于提供的文档提供准确答案的知识助手。
        使用简体中文字符用中文回答。"""
    }
    return prompts.get(lang.lower()[:2], prompts["en"])

def model_res_generator(prompt=""):
    if use_index:
        logging.info(">>> RAG enabled:")
        logging.info(f">>> Query: {prompt}")
        
        # Use the currently selected language for the system prompt
        lang = current_lang.lower()[:2]  # Convert to language code (en, ja, zh)
        system_prompt = get_system_prompt(lang)
        
        retriever = st.session_state.index.as_retriever(similarity_top_k=5)
        retrieved_nodes = retriever.retrieve(prompt)
        
        logging.info(f">>> Retrieved {len(retrieved_nodes)} chunks for context:")
        for i, node in enumerate(retrieved_nodes):
            logging.info(f">>> Chunk {i+1}: Score: {node.score if hasattr(node, 'score') else 'N/A'}; Text: {node.text}")
        
        response_stream = st.session_state.chat_engine.stream_chat(
            prompt,
            context_prompt=system_prompt
        )
        
        for chunk in response_stream.response_gen:
            yield chunk
    else:
        logging.info(">>> Just LLM (no RAG):")
        # Use the currently selected language for the system prompt
        lang = current_lang.lower()[:2]  # Convert to language code (en, ja, zh)
        system_prompt = get_system_prompt(lang)
        
        system_message = {
            "role": "system", 
            "content": system_prompt
        }
        
        messages_only_role_and_content = [system_message] + [
            {"role": message["role"], "content": message["content"]} 
            for message in st.session_state.messages
        ]
        
        try:
            stream = ollama.chat(
                model=st.session_state["model"],
                messages=messages_only_role_and_content,
                stream=True,
            )
            for chunk in stream:
                yield chunk["message"]["content"]
        except Exception as e:
            st.error(f"Error with model {st.session_state['model']}: {str(e)}")
            st.session_state["model"] = "llama3:latest"
            stream = ollama.chat(
                model="llama3:latest",
                messages=messages_only_role_and_content,
                stream=True,
            )
            for chunk in stream:
                yield chunk["message"]["content"]

# Process the prompt if provided
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt, "avatar": AVATAR_USER})
    with st.chat_message("user", avatar=AVATAR_USER):
        st.markdown(prompt)
    with st.chat_message("assistant", avatar=AVATAR_AI):
        with st.spinner("Thinking..."):
            time.sleep(1)
            message = st.write_stream(model_res_generator(prompt))
            st.session_state.messages.append({"role": "assistant", "content": message, "avatar": AVATAR_AI}) 