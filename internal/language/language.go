package language

const (
	CPP      = "cpp"
	GO       = "go"
	JAVA     = "java"
	KOTLIN   = "kotlin"
	JS       = "js"
	TS       = "ts"
	PHP      = "php"
	PROTO    = "proto"
	PYTHON   = "python"
	RST      = "rst"
	RUBY     = "ruby"
	RUST     = "rust"
	SCALA    = "scala"
	SWIFT    = "swift"
	MARKDOWN = "markdown"
	LATEX    = "latex"
	HTML     = "html"
	SOL      = "sol"
	CSHARP   = "csharp"
	COBOL    = "cobol"
)

var FILE_EXTENSIONS = map[string][]string{
	CPP:      {"cpp", "cc", "c", "h", "hpp"},
	GO:       {"go", "templ"},
	JAVA:     {"java"},
	KOTLIN:   {"kt"},
	JS:       {"js", "jsx", "cjs", "mjs"},
	TS:       {"ts", "tsx"},
	PHP:      {"php"},
	PROTO:    {"proto"},
	PYTHON:   {"py"},
	RST:      {"rst"},
	RUBY:     {"rb"},
	RUST:     {"rs"},
	SCALA:    {"scala"},
	SWIFT:    {"swift"},
	MARKDOWN: {"md"},
	LATEX:    {"tex"},
	HTML:     {"html", "cshtml"},
	SOL:      {"sol"},
	CSHARP:   {"cs", "cshtml"},
	COBOL:    {"cbl"},
}
