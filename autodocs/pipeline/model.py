import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv("/Users/edenhyacinth/autodocs/.env")

from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel


class ChatModel:
    _model: Optional[BaseChatModel] = None

    @staticmethod
    def load_model() -> BaseChatModel:
        return ChatOpenAI(
            model_name=os.environ.get("LLM_BASE_MODEL_NAME", "gpt-3.5-turbo-0613")
        )

    @classmethod
    def model(cls) -> BaseChatModel:
        if cls._model is None:
            cls._model = cls.load_model()
        assert cls._model is not None
        return cls._model
