package embeddings

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/j4ndrw/codebase-indexer/internal/constant"
	"github.com/tmc/langchaingo/embeddings"
)

type Embedder struct{}

var _ embeddings.Embedder = &Embedder{}

func New() (*Embedder, error) {
	return &Embedder{}, nil
}

func (embedder *Embedder) EmbedDocuments(ctx context.Context, texts []string) ([][]float32, error) {
	body := map[string][]string{"input": texts}
	data, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	res, err := http.Post(
		fmt.Sprintf("%s/predictions/my_model", constant.TORCH_SERVE_HOST),
		"application/json",
		bytes.NewBuffer(data),
	)
	if err != nil {
		return nil, err
	}

	defer res.Body.Close()

	decoder := json.NewDecoder(res.Body)

	var embeds [][]float32
	if err := decoder.Decode(&embeds); err != nil {
		return nil, err
	}
	return embeds, nil
}

func (embedder *Embedder) EmbedQuery(ctx context.Context, text string) ([]float32, error) {
	body := map[string][]string{"input": {text}}
	data, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	res, err := http.Post(
		fmt.Sprintf("%s/predictions/my_model", constant.TORCH_SERVE_HOST),
		"application/json",
		strings.NewReader(string(data)),
	)
	if err != nil {
		return nil, err
	}

	defer res.Body.Close()

	decoder := json.NewDecoder(res.Body)

	var embeds [][]float32
	if err := decoder.Decode(&embeds); err != nil {
		return nil, err
	}
	return embeds[0], nil
}
