import os
import pathlib
import langchain
from typing import Optional

from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel

from langchain.cache import SQLiteCache

# TODO: Replace with shared Cache
autodocs_path = pathlib.Path.home() / ".autodocs"
autodocs_path.mkdir(exist_ok=True)
langchain.llm_cache = SQLiteCache(
    database_path=str(autodocs_path / "default_langchain_cache.db")
)


class ChatModel:
    _model: Optional[BaseChatModel] = None

    @staticmethod
    def load_model() -> BaseChatModel:
        return ChatOpenAI(model_name=os.environ.get("LLM_BASE_MODEL_NAME", "gpt-4"))

    @staticmethod
    def load_3_5_model() -> BaseChatModel:
        return ChatOpenAI(model_name="gpt-3.5-turbo")

    @classmethod
    def model(cls) -> BaseChatModel:
        if cls._model is None:
            cls._model = cls.load_model()
        assert cls._model is not None
        return cls._model
