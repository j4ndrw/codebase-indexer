package utils

import (
	"fmt"
	"strings"
)

func StripGeneratedText(s string) string {
	s = strings.Replace(s, "<|im_end|>", "", -1)
	s = strings.Replace(s, "'", "", -1)
	s = strings.Replace(s, "\"", "", -1)
	s = strings.TrimSpace(s)
	return s
}

func SliceContains[T comparable](s []T, e T) bool {
	for _, a := range s {
		if a == e {
			return true
		}
	}
	return false
}

func CollectionName(s string) string {
	split := strings.Split(s, "/")
	s = func() string {
		switch len(split) {
		case 0:
			return s
		case 1:
			return split[0]
		default:
			return fmt.Sprintf("%s_%s", split[len(split)-2], split[len(split)-1])
		}
	}()
	s = strings.Replace(s, "/", "_", -1)
	s = strings.Replace(s, "-", "_", -1)
	return s
}
