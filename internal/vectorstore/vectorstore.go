package vectorstore

import "github.com/amikos-tech/chroma-go"

type VectorStore struct {
	client     *chroma.Client
	collection *chroma.Collection
}

func New(
	client *chroma.Client,
	collection *chroma.Collection,
) *VectorStore {
	return &VectorStore{client, collection}
}
