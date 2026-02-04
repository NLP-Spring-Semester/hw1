# src/part1_regex.py
import re

# Replace tokens
MENTION_TOKEN = "[MENTION]"
URL_TOKEN = "[URL]"
HASHTAG_TOKEN = "[HASHTAG]"

# @switchfoot, @Kenichan, @angry barista, @Alliana07
# Match @word, optionally capturing a second word only when more text follows.
# The lookahead (?=\s+\w) ensures we don't greedily consume words that border
# non-word tokens (like [URL] after pipeline replacement).
MENTION_RE = re.compile(r"@\w+(?:\s+\w+(?=\s+\w))?")

# Matches URLs
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

# Matches the hashtag and the word following it.
HASHTAG_RE = re.compile(r"#\w+")


def replace_mentions(text: str) -> str:
    return MENTION_RE.sub(MENTION_TOKEN, text)


def replace_urls(text: str) -> str:
    return URL_RE.sub(URL_TOKEN, text)


def replace_hashtags(text: str) -> str:
    return HASHTAG_RE.sub(HASHTAG_TOKEN, text)


def preprocess_part1(text: str) -> str:
    """
    Combined Function call
    """
    text = replace_urls(text)
    text = replace_mentions(text)
    text = replace_hashtags(text)
    return text
