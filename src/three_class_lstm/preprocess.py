import re

URL_PATTERN = re.compile(r"http\S+|www\S+")
USER_PATTERN = re.compile(r"@\w+")

def clean_text(text: str) -> str:
    text = text.lower()
    text = URL_PATTERN.sub("<URL>", text)
    text = USER_PATTERN.sub("<USER>", text)
    return text.strip()
