package documentloaders

import (
	"context"
	"io"
	"os"
	"path/filepath"

	"github.com/go-git/go-git/v5"
	"github.com/go-git/go-git/v5/plumbing/object"

	"github.com/tmc/langchaingo/documentloaders"
	"github.com/tmc/langchaingo/schema"
	"github.com/tmc/langchaingo/textsplitter"
)

type GitLoader struct {
	repoPath   string
	fileFilter *func(string) bool
}

var _ documentloaders.Loader = &GitLoader{}

func NewGitLoader(repoPath string, fileFilter *func(string) bool) *GitLoader {
	return &GitLoader{
		repoPath:   repoPath,
		fileFilter: fileFilter,
	}
}

func (loader *GitLoader) readFromBlob(blob io.ReadCloser) (string, error) {
	defer blob.Close()

	buffer := make([]byte, 1024)
	content := ""
	for {
		n, err := blob.Read(buffer)
		if err != nil && err != io.EOF {
			return "", err
		}
		if n == 0 {
			break
		}
		content += string(buffer)
	}
	return content, nil
}

func (loader *GitLoader) getRepo() (*git.Repository, error) {
	repo, err := git.PlainOpen(loader.repoPath)
	if err != nil {
		return nil, err
	}
	return repo, nil
}

func (loader *GitLoader) getCommit() (*object.Commit, error) {
	repo, err := loader.getRepo()
	if err != nil {
		return nil, err
	}

	head, err := repo.Head()
	if err != nil {
		return nil, err
	}

	commit, err := repo.CommitObject(head.Hash())
	if err != nil {
		return nil, err
	}

	return commit, nil
}

func (loader *GitLoader) loadFilePaths() (map[string]object.Blob, error) {
	commit, err := loader.getCommit()
	if err != nil {
		return nil, err
	}
	fileMap := map[string]object.Blob{}

	files, err := commit.Files()
	if err != nil {
		return nil, err
	}
	for {
		file, err := files.Next()
		if err != nil {
			break
		}
		path := filepath.Join(loader.repoPath, filepath.FromSlash(file.Name))

		if loader.fileFilter != nil {
			fileFilter := *loader.fileFilter
			if !fileFilter(path) {
				continue
			}
		}
		fileMap[path] = file.Blob
	}
	return fileMap, nil
}

func (loader *GitLoader) readFile(filePath string) (string, error) {
	content, err := os.ReadFile(filePath)
	if err != nil {
		return "", err
	}
	return string(content), err
}

func (loader *GitLoader) Load(ctx context.Context) ([]schema.Document, error) {
	if _, err := os.Stat(loader.repoPath); os.IsNotExist(err) {
		return nil, err
	}

	fileMap, err := loader.loadFilePaths()
	if err != nil {
		return nil, err
	}

	var docs []schema.Document
	for filePath, blob := range fileMap {
		reader, err := blob.Reader()
		if err != nil {
			return nil, err
		}
		content, err := loader.readFromBlob(reader)
		if err != nil {
			return nil, err
		}

		fileType := filepath.Ext(filePath)
		metadata := map[string]any{
			"source":   filePath,
			"fileType": fileType,
		}
		doc := schema.Document{
			PageContent: content,
			Metadata:    metadata,
		}
		docs = append(docs, doc)
	}
	return docs, nil
}

func (loader *GitLoader) LoadAndSplit(ctx context.Context, splitter textsplitter.TextSplitter) ([]schema.Document, error) {
	docs, err := loader.Load(ctx)
	if err != nil {
		return nil, err
	}

	var splitDocs []schema.Document
	for _, doc := range docs {
		texts, err := splitter.SplitText(doc.PageContent)
		if err != nil {
			return nil, err
		}
		for _, text := range texts {
			splitDocs = append(splitDocs, schema.Document{
				PageContent: text,
				Metadata:    doc.Metadata,
				Score:       doc.Score,
			})
		}
	}

	return splitDocs, nil
}
