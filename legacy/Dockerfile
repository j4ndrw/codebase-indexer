ARG OLLAMA_HOST=127.0.0.1:11434

FROM python:3.10

ENV HOME /home/codebase_indexer
ENV PROJECT_DIR ${HOME}/codebase_indexer
ENV PYTHONPATH ${PROJECT_DIR}}
ENV OLLAMA_HOST $OLLAMA_HOST

RUN useradd -ms /bin/bash codebase_indexer
RUN usermod -aG sudo codebase_indexer

USER codebase_indexer

RUN pip install --user pipenv

WORKDIR ${PROJECT_DIR}
COPY . ${PROJECT_DIR}/

RUN python -m pipenv install --system --deploy

EXPOSE 11435
ENTRYPOINT ["python", "main.py"]
