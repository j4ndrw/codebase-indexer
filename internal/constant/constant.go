package constant

import (
	"fmt"
	"os"
)

var OLLAMA_BASE_URL string = fmt.Sprintf("http://%s", os.Getenv("OLLAMA_HOST"))
var DEFAULT_VECTOR_DB_DIR string = func() string {
	homeDir, err := os.UserHomeDir()
	if err != nil {
		panic(err)
	}
	return fmt.Sprintf("%s/.codebase-indexer", homeDir)
}()

const DEFAULT_OLLAMA_INFERENCE_MODEL = "mistral-openorca:7b"
const MAX_SOURCES_WINDOW = 10
