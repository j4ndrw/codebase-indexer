package command

const (
	TEST_COMMAND = iota
	SEARCH_COMMAND
	REVIEW_COMMAND
	GENERAL_CHAT_COMMAND
	NEW_CHAT_COMMAND
)

var commands = []string{"test", "search", "review", "general_chat", "new_chat"}

func Get(command int) string {
	return commands[command]
}
