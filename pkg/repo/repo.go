package repo

import "github.com/go-git/go-git"

type Repo struct {
	Path string
	Repo *git.Repository
}

func New(path string) (*Repo, error) {
	repo, err := git.PlainOpen(path)
	if err != nil {
		return nil, err
	}

	return &Repo{
		Path: path,
		Repo: repo,
	}, nil
}
