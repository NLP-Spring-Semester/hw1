import regex as re
from collections import Counter, defaultdict


def space_tokenize(text):
    """
    Simple space-based tokenization.
    """
    return text.split()


# This pattern is taken from the textbook and simplified into a singel regex.
GPT2_PATTERN = re.compile(
    r"'s|'t|'re|'ve|'m|'ll|'d|"
    r" ?\p{L}+|"
    r" ?\p{N}+|"
    r" ?[^\s\p{L}\p{N}]+|"
    r"\s+(?!\S)|\s+"
)


def gpt2_pretokenize(text):
    """
    GPT-2 pretokenization that splits on word boundaries.
    """
    return GPT2_PATTERN.findall(text)


def get_byte_pairs(word):
    """
    Get all adjacent byte pairs in a word.
    """
    pairs = set()
    for i in range(len(word) - 1):
        pairs.add((word[i], word[i + 1]))
    return pairs


def merge_pair(word, pair):
    """
    Merge a specific pair in a word.
    """
    new_word = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
            new_word.append(pair[0] + pair[1])
            i += 2
        else:
            new_word.append(word[i])
            i += 1
    return new_word


class BPETokenizer:
    """
    Byte Pair Encoding tokenizer with GPT-2 style pretokenization.
    """

    def __init__(self, num_merges=1000):
        self.num_merges = num_merges
        self.merges = []
        self.vocab = {}

    def train(self, corpus):
        """
        Train the BPE tokenizer on the corpus that we provide.
        """
        # Pretokenize corpus
        words = []
        for text in corpus:
            words.extend(gpt2_pretokenize(text))

        # Convert to bytes
        word_freqs = Counter(words)
        splits = {}
        for word, freq in word_freqs.items():
            byte_list = list(word.encode("utf-8"))
            # Convert bytes to separate tokens initially
            splits[word] = [
                bytes([b]).decode("utf-8", errors="replace") for b in byte_list
            ]

        # Do our merges
        for _ in range(self.num_merges):
            # Count all pairs
            pair_freqs = defaultdict(int)
            for word, freq in word_freqs.items():
                split = splits[word]
                pairs = get_byte_pairs(split)
                for pair in pairs:
                    pair_freqs[pair] += freq

            if not pair_freqs:
                break

            # Find most frequent pair
            best_pair = max(pair_freqs, key=pair_freqs.get)
            self.merges.append(best_pair)

            # Merge the pair in all words
            for word in splits:
                splits[word] = merge_pair(splits[word], best_pair)

        # Build vocabulary
        vocab_set = set()
        for word in splits.values():
            vocab_set.update(word)
        self.vocab = {token: i for i, token in enumerate(sorted(vocab_set))}

    def tokenize(self, text):
        """
        Tokenize the text using learned merges.
        """
        words = gpt2_pretokenize(text)
        tokens = []

        for word in words:
            # Convert to bytes as that is our input format
            byte_list = list(word.encode("utf-8"))
            split = [bytes([b]).decode("utf-8", errors="replace") for b in byte_list]

            # Apply merges in order
            for merge in self.merges:
                split = merge_pair(split, merge)

            tokens.extend(split)

        return tokens


class SentencePieceBPE:
    """
    SentencePiece variant of BPE - no pretokenization, treats entire text as character sequence.
    """

    def __init__(self, num_merges=1000):
        self.num_merges = num_merges
        self.merges = []
        self.vocab = {}

    def train(self, corpus):
        # Concatenate all text as there is no pretokenization
        full_text = " ".join(corpus)
        byte_list = list(full_text.encode("utf-8"))
        splits = {}

        # the entire corpus is diviced into chunks to make training easier.
        chunk_size = 1000
        chunks = [
            full_text[i : i + chunk_size] for i in range(0, len(full_text), chunk_size)
        ]

        for chunk in chunks:
            byte_list = list(chunk.encode("utf-8"))
            splits[chunk] = [
                bytes([b]).decode("utf-8", errors="replace") for b in byte_list
            ]

        chunk_freqs = {chunk: 1 for chunk in chunks}

        # Perform merges
        for _ in range(self.num_merges):
            # Count all pairs
            pair_freqs = defaultdict(int)
            for chunk, freq in chunk_freqs.items():
                split = splits[chunk]
                pairs = get_byte_pairs(split)
                for pair in pairs:
                    pair_freqs[pair] += freq

            if not pair_freqs:
                break

            # Find most frequent pair
            best_pair = max(pair_freqs, key=pair_freqs.get)
            self.merges.append(best_pair)

            # Merge the pair in all chunks
            for chunk in splits:
                splits[chunk] = merge_pair(splits[chunk], best_pair)

        # Build vocabulary
        vocab_set = set()
        for chunk in splits.values():
            vocab_set.update(chunk)
        self.vocab = {token: i for i, token in enumerate(sorted(vocab_set))}

    def tokenize(self, text):
        # Convert to bytes
        byte_list = list(text.encode("utf-8"))
        split = [bytes([b]).decode("utf-8", errors="replace") for b in byte_list]

        # Apply merges in order
        for merge in self.merges:
            split = merge_pair(split, merge)

        return split


if __name__ == "__main__":
    # Example usage
    test_corpus = [
        "This is a sentence.",
        "This is another sentence.",
        "Tokenization is fun!",
    ]

    print("Space-based tokenization:")
    for text in test_corpus:
        print(f"{text} -> {space_tokenize(text)}")

    print("\nBPE tokenization:")
    bpe = BPETokenizer(num_merges=10)
    bpe.train(test_corpus)
    for text in test_corpus[:1]:
        print(f"{text} -> {bpe.tokenize(text)}")

    print("\nSentencePiece BPE tokenization:")
    sp_bpe = SentencePieceBPE(num_merges=10)
    sp_bpe.train(test_corpus)
    for text in test_corpus[:1]:
        print(f"{text} -> {sp_bpe.tokenize(text)}")
