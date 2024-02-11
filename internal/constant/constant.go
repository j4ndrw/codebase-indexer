package constant

import (
	"fmt"
	"os"
)

var OLLAMA_BASE_URL string = fmt.Sprintf("http://%s", func() string {
	url := os.Getenv("OLLAMA_HOST")
	if url == "" {
		return "localhost:11434"
	}
	return url
}())

var CHROMA_HOST string = fmt.Sprintf("http://%s", func() string {
	url := os.Getenv("CHROMA_HOST")
	if url == "" {
		return "localhost:8000"
	}
	return url
}())

var TORCH_SERVE_HOST string = fmt.Sprintf("http://%s", func() string {
	url := os.Getenv("TORCH_SERVE_HOST")
	if url == "" {
		return "localhost:8080"
	}
	return url
}())

var DEFAULT_VECTOR_DB_DIR string = func() string {
	homeDir, err := os.UserHomeDir()
	if err != nil {
		panic(err)
	}
	return fmt.Sprintf("%s/.codebase-indexer", homeDir)
}()

const DEFAULT_OLLAMA_INFERENCE_MODEL = "mistral-openorca:7b"
const MAX_SOURCES_WINDOW = 10
