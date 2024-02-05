from typing import Literal

from pydantic import BaseModel


class MetaSchema(BaseModel):
    repo_path: str
    sub_folder: str | None = None
    vector_db_dir: str | None = None
    ollama_inference_model: str | None = None
    kind: Literal["indexer", "completion"] | None = None

class CompleteSchema(BaseModel):
    current_buffer_pre: str
    current_buffer_suf: str
    related_buffers: list[str]

Command = Literal["test", "search", "review", "forget_previous_conversation", "general_chat"]
