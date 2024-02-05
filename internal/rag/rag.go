package rag

import (
	"context"
	"strings"

	"github.com/amikos-tech/chroma-go"
	"github.com/go-git/go-git"
	"github.com/j4ndrw/codebase-indexer/internal/constant"
	"github.com/j4ndrw/codebase-indexer/internal/vectorstore"
	"github.com/j4ndrw/codebase-indexer/pkg/repo"
	"github.com/tmc/langchaingo/llms/ollama"
)

func CreateLLM(model *string) *ollama.LLM {
	if model == nil {
		defaultModel := constant.DEFAULT_OLLAMA_INFERENCE_MODEL
		model = &defaultModel
	}
	llm, err := ollama.New(
		ollama.WithModel(*model),
		ollama.WithServerURL(constant.OLLAMA_BASE_URL),
	)
	if err != nil {
		panic(err)
	}
	return llm
}

func CreateVectorStore(ctx *context.Context, path *string, repo repo.Repo) *vectorstore.VectorStore {
	if path == nil {
		defaultPath := constant.DEFAULT_VECTOR_DB_DIR
		path = &defaultPath
	}
	client := chroma.NewClient(*path)
	collection, err := client.CreateCollection(ctx, strings.Replace(repo.Path, "/", "_"), nil, nil)
	if err != nil {
		panic(err)
	}
	return vectorstore.New(client, collection)
}
