from typing import Literal

from pydantic import BaseModel


class Meta(BaseModel):
    repo_path: str
    sub_folder: str | None = None
    vector_db_dir: str | None = None
    ollama_inference_model: str | None = None


Command = Literal["test", "explain", "search", "doc", "fix", "review"]
