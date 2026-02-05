import csv
import os
import time
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.part1_regex import preprocess_part1
from src.part2_tokenization import space_tokenize, BPETokenizer, SentencePieceBPE

SENTIMENT_PATH = "datasets/sentiment140_noemoticon_10000.csv"
WIKI_PATH = "datasets/simple_english_wikipedia_10000.txt"
OUTPUT_DIR = "output"
NUM_MERGES = 1000


def load_sentiment140():
    texts = []
    with open(SENTIMENT_PATH, encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        for row in reader:
            texts.append(row[5])
    return texts


def load_wikipedia():
    texts = []
    with open(WIKI_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                texts.append(line)
    return texts


def get_vocab_and_freqs(tokenized_texts):
    counter = Counter()
    for tokens in tokenized_texts:
        counter.update(tokens)
    vocab = set(counter.keys())
    return vocab, counter


def time_tokenization(fn, texts):
    start = time.time()
    results = [fn(t) for t in texts]
    elapsed = time.time() - start
    return elapsed, results


def save_freq_plot(freqs, title, filename, top_n=50):
    most_common = freqs.most_common(top_n)
    labels = [tok for tok, _ in most_common]
    counts = [c for _, c in most_common]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(range(len(counts)), counts)
    ax.set_title(title)
    ax.set_ylabel("Frequency")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close()


def write(f, text=""):
    print(text)
    f.write(text + "\n")


def write_table(f, title, rows, headers):
    write(f, f"\n{'='*60}")
    write(f, title)
    write(f, f"{'='*60}")
    write(f, "  ".join(h.ljust(30) for h in headers))
    write(f, "-" * (32 * len(headers)))
    for row in rows:
        write(f, "  ".join(str(c).ljust(30) for c in row))


def write_token_list(f, title, tokens):
    write(f, f"\n--- {title} ---")
    for i, (tok, count) in enumerate(tokens):
        write(f, f"  {i+1:4d}. {repr(tok):30s}  freq={count}")


def run_comparison_1(f, corpus_name, texts):
    write(f, f"\n{'#'*60}")
    write(f, f"# Analysis 1: Space vs BPE — {corpus_name}")
    write(f, f"{'#'*60}")

    # Space tokenization
    t_space, space_results = time_tokenization(space_tokenize, texts)
    space_vocab, space_freqs = get_vocab_and_freqs(space_results)

    # BPE tokenization
    bpe = BPETokenizer(num_merges=NUM_MERGES)
    t_train_start = time.time()
    bpe.train(texts)
    t_train = time.time() - t_train_start
    t_bpe, bpe_results = time_tokenization(bpe.tokenize, texts)
    bpe_vocab, bpe_freqs = get_vocab_and_freqs(bpe_results)

    tag = corpus_name.lower().replace(" ", "_")

    write_table(f, f"Summary — {corpus_name}", [
        ["Space", len(space_vocab), f"{t_space:.3f}s", "N/A"],
        ["BPE", len(bpe_vocab), f"{t_bpe:.3f}s", f"{t_train:.3f}s"],
    ], ["Method", "Vocab Size", "Tokenize Time", "Train Time"])

    write_token_list(f, f"Space Top 100 — {corpus_name}", space_freqs.most_common(100))
    write_token_list(f, f"Space Bottom 100 — {corpus_name}", space_freqs.most_common()[-(100):])
    write_token_list(f, f"BPE Top 100 — {corpus_name}", bpe_freqs.most_common(100))
    write_token_list(f, f"BPE Bottom 100 — {corpus_name}", bpe_freqs.most_common()[-(100):])

    longest_bpe = sorted(bpe_vocab, key=len, reverse=True)[:100]
    write(f, f"\n--- BPE Longest 100 Subwords — {corpus_name} ---")
    for i, tok in enumerate(longest_bpe):
        write(f, f"  {i+1:4d}. {repr(tok):40s}  len={len(tok)}")

    save_freq_plot(space_freqs, f"Space Top-50 Frequencies — {corpus_name}", f"1_space_freq_{tag}.png")
    save_freq_plot(bpe_freqs, f"BPE Top-50 Frequencies — {corpus_name}", f"1_bpe_freq_{tag}.png")


def run_comparison_2(f, texts):
    write(f, f"\n{'#'*60}")
    write(f, "# Analysis 2: BPE With vs Without Regex — Sentiment140")
    write(f, f"{'#'*60}")

    texts_clean = [preprocess_part1(t) for t in texts]

    # BPE without regex
    bpe_raw = BPETokenizer(num_merges=NUM_MERGES)
    t_train_raw_start = time.time()
    bpe_raw.train(texts)
    t_train_raw = time.time() - t_train_raw_start
    t_raw, raw_results = time_tokenization(bpe_raw.tokenize, texts)
    raw_vocab, raw_freqs = get_vocab_and_freqs(raw_results)

    # BPE with regex
    bpe_clean = BPETokenizer(num_merges=NUM_MERGES)
    t_train_clean_start = time.time()
    bpe_clean.train(texts_clean)
    t_train_clean = time.time() - t_train_clean_start
    t_clean, clean_results = time_tokenization(bpe_clean.tokenize, texts_clean)
    clean_vocab, clean_freqs = get_vocab_and_freqs(clean_results)

    write_table(f, "Summary — Sentiment140", [
        ["BPE (no regex)", len(raw_vocab), f"{t_raw:.3f}s", f"{t_train_raw:.3f}s"],
        ["BPE (with regex)", len(clean_vocab), f"{t_clean:.3f}s", f"{t_train_clean:.3f}s"],
    ], ["Method", "Vocab Size", "Tokenize Time", "Train Time"])

    write_token_list(f, "BPE No-Regex Top 100", raw_freqs.most_common(100))
    write_token_list(f, "BPE No-Regex Bottom 100", raw_freqs.most_common()[-(100):])
    write_token_list(f, "BPE With-Regex Top 100", clean_freqs.most_common(100))
    write_token_list(f, "BPE With-Regex Bottom 100", clean_freqs.most_common()[-(100):])

    longest_raw = sorted(raw_vocab, key=len, reverse=True)[:100]
    longest_clean = sorted(clean_vocab, key=len, reverse=True)[:100]
    write(f, "\n--- BPE No-Regex Longest 100 Subwords ---")
    for i, tok in enumerate(longest_raw):
        write(f, f"  {i+1:4d}. {repr(tok):40s}  len={len(tok)}")
    write(f, "\n--- BPE With-Regex Longest 100 Subwords ---")
    for i, tok in enumerate(longest_clean):
        write(f, f"  {i+1:4d}. {repr(tok):40s}  len={len(tok)}")

    save_freq_plot(raw_freqs, "BPE No-Regex Top-50 — Sentiment140", "2_bpe_noregex_freq.png")
    save_freq_plot(clean_freqs, "BPE With-Regex Top-50 — Sentiment140", "2_bpe_regex_freq.png")


def run_comparison_3(f, corpus_name, texts):
    write(f, f"\n{'#'*60}")
    write(f, f"# Analysis 3: BPE vs SentencePiece — {corpus_name}")
    write(f, f"{'#'*60}")

    # BPE
    bpe = BPETokenizer(num_merges=NUM_MERGES)
    t_train_bpe_start = time.time()
    bpe.train(texts)
    t_train_bpe = time.time() - t_train_bpe_start
    t_bpe, bpe_results = time_tokenization(bpe.tokenize, texts)
    bpe_vocab, bpe_freqs = get_vocab_and_freqs(bpe_results)

    # SentencePiece
    sp = SentencePieceBPE(num_merges=NUM_MERGES)
    t_train_sp_start = time.time()
    sp.train(texts)
    t_train_sp = time.time() - t_train_sp_start
    t_sp, sp_results = time_tokenization(sp.tokenize, texts)
    sp_vocab, sp_freqs = get_vocab_and_freqs(sp_results)

    tag = corpus_name.lower().replace(" ", "_")

    write_table(f, f"Summary — {corpus_name}", [
        ["BPE", len(bpe_vocab), f"{t_bpe:.3f}s", f"{t_train_bpe:.3f}s"],
        ["SentencePiece", len(sp_vocab), f"{t_sp:.3f}s", f"{t_train_sp:.3f}s"],
    ], ["Method", "Vocab Size", "Tokenize Time", "Train Time"])

    write_token_list(f, f"BPE Top 100 — {corpus_name}", bpe_freqs.most_common(100))
    write_token_list(f, f"BPE Bottom 100 — {corpus_name}", bpe_freqs.most_common()[-(100):])
    write_token_list(f, f"SentencePiece Top 100 — {corpus_name}", sp_freqs.most_common(100))
    write_token_list(f, f"SentencePiece Bottom 100 — {corpus_name}", sp_freqs.most_common()[-(100):])

    longest_bpe = sorted(bpe_vocab, key=len, reverse=True)[:100]
    longest_sp = sorted(sp_vocab, key=len, reverse=True)[:100]
    write(f, f"\n--- BPE Longest 100 Subwords — {corpus_name} ---")
    for i, tok in enumerate(longest_bpe):
        write(f, f"  {i+1:4d}. {repr(tok):40s}  len={len(tok)}")
    write(f, f"\n--- SentencePiece Longest 100 Subwords — {corpus_name} ---")
    for i, tok in enumerate(longest_sp):
        write(f, f"  {i+1:4d}. {repr(tok):40s}  len={len(tok)}")

    save_freq_plot(bpe_freqs, f"BPE Top-50 Frequencies — {corpus_name}", f"3_bpe_freq_{tag}.png")
    save_freq_plot(sp_freqs, f"SentencePiece Top-50 Frequencies — {corpus_name}", f"3_sp_freq_{tag}.png")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading datasets...")
    sent_texts = load_sentiment140()
    wiki_texts = load_wikipedia()
    print(f"  Sentiment140: {len(sent_texts)} texts")
    print(f"  Wikipedia:    {len(wiki_texts)} texts")

    out_path = os.path.join(OUTPUT_DIR, "analysis_data.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        # Analysis 1
        run_comparison_1(f, "Sentiment140", sent_texts)
        run_comparison_1(f, "Wikipedia", wiki_texts)

        # Analysis 2
        run_comparison_2(f, sent_texts)

        # Analysis 3
        run_comparison_3(f, "Sentiment140", sent_texts)
        run_comparison_3(f, "Wikipedia", wiki_texts)

    print(f"\nAll outputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
