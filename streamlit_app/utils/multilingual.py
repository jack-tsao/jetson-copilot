from typing import List, Dict
import logging
from langdetect import detect
from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding

class MultilingualRAG:
    def __init__(self, use_openai: bool = False):
        self.use_openai = use_openai
        if use_openai:
            self.embed_model = OpenAIEmbedding()
        else:
            self.embed_model = OllamaEmbedding("mxbai-embed-large:latest")
        Settings.embed_model = self.embed_model

    def detect_language(self, text: str) -> str:
        """Detect the language of the input text."""
        try:
            return detect(text)
        except:
            return "en"

    def get_system_prompt(self, lang: str) -> str:
        """Get the system prompt in the detected language."""
        prompts = {
            "en": """You are a precise knowledge assistant that provides accurate answers based solely on the provided documents.
            Respond in English using standard English characters.""",
            "ja": """あなたは提供された文書に基づいて正確な回答を提供する知識アシスタントです。
            ひらがな、カタカナ、漢字を使用して日本語で回答してください。""",
            "zh": """你是一个基于提供的文档提供准确答案的知识助手。
            使用简体中文字符用中文回答。"""
        }
        return prompts.get(lang, prompts["en"])

    def process_query(self, query: str, context: str) -> Dict:
        """Process a query and return the necessary components for RAG."""
        lang = self.detect_language(query)
        system_prompt = self.get_system_prompt(lang)
        
        return {
            "query": query,
            "context": context,
            "system_prompt": system_prompt,
            "language": lang
        }

    def format_response(self, response: str, lang: str) -> str:
        """Format the response based on the detected language."""
        # Add any language-specific formatting here
        return response 