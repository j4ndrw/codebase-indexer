def strip_generated_text(s: str) -> str:
    return s.replace("<|im_end|>", "").replace("'", "").replace('"', "").strip()
