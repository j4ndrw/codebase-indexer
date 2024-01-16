ARG OLLAMA_HOST=127.0.0.1:11434
FROM python:3.10

RUN pip install pipenv

ENV PROJECT_DIR /llm-server
ENV OLLAMA_HOST $OLLAMA_HOST

WORKDIR ${PROJECT_DIR}

COPY . ${PROJECT_DIR}/

RUN pipenv install --system --deploy

ENV PYTHONPATH /llm-server/

EXPOSE 11435
CMD ["uvicorn", "main:app", "--port", "11435"]
