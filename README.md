# Codebase Indexer

## Prerequisites

1. Chroma DB:

```console
$ docker pull chromadb/chroma
$ docker run -p 8000:8000 chromadb/chroma
```

2. Torch serve:

```console
$ docker run -d -p 8080:8080 -it ghcr.io/clems4ever/torchserve-all-minilm-l6-v2:latest
```
