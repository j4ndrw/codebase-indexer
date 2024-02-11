package main

import (
	"context"
	"fmt"
	"path/filepath"
	"strings"

	"github.com/j4ndrw/codebase-indexer/internal/constant"
	"github.com/j4ndrw/codebase-indexer/internal/documentloaders"
	"github.com/j4ndrw/codebase-indexer/internal/embeddings"
	"github.com/j4ndrw/codebase-indexer/internal/utils"
	"github.com/tmc/langchaingo/textsplitter"
	"github.com/tmc/langchaingo/vectorstores/chroma"
)

func main() {
	ctx := context.Background()

	query := "where are the languages defined"

	repoPath := filepath.FromSlash("/home/j4ndrw/pp/codebase-indexer")

	embedder, err := embeddings.New()
	if err != nil {
		panic(err)
	}
	vectorStore, err := chroma.New(
		chroma.WithEmbedder(embedder),
		chroma.WithNameSpace(utils.CollectionName(repoPath)),
		chroma.WithChromaURL(constant.CHROMA_HOST),
	)
	if err != nil {
		panic(err)
	}

	fileFilter := func(path string) bool {
		for _, pathToExclude := range []string{
			"node_modules",
			".git",
			"vendor",
			"dist",
			"build",
			"go.sum",
			"go.mod",
			"package-lock.json",
			"pnpm-lock.yaml",
			"pnpm-lock.yml",
			"yarn.lock",
		} {
			if strings.Contains(path, pathToExclude) {
				return false
			}
		}
		return true
	}
	gitLoader := documentloaders.NewGitLoader(repoPath, &fileFilter)
	recursiveTextSplitter := textsplitter.NewRecursiveCharacter(textsplitter.WithChunkOverlap(200), textsplitter.WithChunkSize(2000))
	documents, err := gitLoader.LoadAndSplit(ctx, recursiveTextSplitter)
	if err != nil {
		panic(err)
	}

	if _, err := vectorStore.AddDocuments(ctx, documents); err != nil {
		panic(err)
	}
	foundDocs, err := vectorStore.SimilaritySearch(ctx, query, 10)
	if err != nil {
		panic(err)
	}
	fmt.Println(foundDocs[0])
}
