package utils

import "strings"

func StripGeneratedText(s string) string {
	s = strings.Replace(s, "<|im_end|>", "", -1)
	s = strings.Replace(s, "'", "", -1)
	s = strings.Replace(s, "\"", "", -1)
	s = strings.TrimSpace(s)
	return s
}
