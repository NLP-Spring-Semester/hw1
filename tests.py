from src.part1_regex import (
    replace_mentions,
    replace_urls,
    replace_hashtags,
    preprocess_part1,
    MENTION_TOKEN,
    URL_TOKEN,
    HASHTAG_TOKEN,
)
from src.part2_tokenization import (
    BPETokenizer,
    SentencePieceBPE,
)


def check(name, got, expected):
    ok = got == expected
    print(f"- {name}: {'PASS' if ok else 'FAIL'}")
    if not ok:
        print("  got     :", repr(got))
        print("  expected:", repr(expected))
    return ok


def main():
    print("Part 1 tests: regex-based replacements\n")

    passed = 0
    total = 0

    # Mentions
    total += 1
    passed += check(
        "mention basic", replace_mentions("hi @Kenichan"), f"hi {MENTION_TOKEN}"
    )

    total += 1
    passed += check(
        "mention with space (example style)",
        replace_mentions("talk to @angry barista now"),
        f"talk to {MENTION_TOKEN} now",
    )

    total += 1
    passed += check(
        "mention at start of string",
        replace_mentions("@switchfoot how are you"),
        f"{MENTION_TOKEN} are you",
    )

    total += 1
    passed += check(
        "mention at end of string",
        replace_mentions("shoutout to @Alliana07"),
        f"shoutout to {MENTION_TOKEN}",
    )

    total += 1
    passed += check(
        "mention with underscore",
        replace_mentions("follow @cool_user_99 please"),
        f"follow {MENTION_TOKEN} please",
    )

    total += 1
    passed += check(
        "mention with numbers",
        replace_mentions("ask @user123 about it"),
        f"ask {MENTION_TOKEN} it",
    )

    total += 1
    passed += check(
        "multiple mentions",
        replace_mentions("@alice and @bob are here"),
        f"{MENTION_TOKEN} and {MENTION_TOKEN} here",
    )

    total += 1
    passed += check(
        "mention followed by punctuation",
        replace_mentions("thanks @Kenichan!"),
        f"thanks {MENTION_TOKEN}!",
    )

    total += 1
    passed += check(
        "mention followed by comma",
        replace_mentions("hey @alice, what's up"),
        f"hey {MENTION_TOKEN}, what's up",
    )

    total += 1
    passed += check(
        "no mentions passthrough",
        replace_mentions("just a normal sentence"),
        "just a normal sentence",
    )

    total += 1
    passed += check(
        "empty string mentions",
        replace_mentions(""),
        "",
    )

    total += 1
    passed += check(
        "bare @ not a mention",
        replace_mentions("email me @ noon"),
        "email me @ noon",
    )

    # URLs
    total += 1
    passed += check(
        "url http",
        replace_urls("link http://twitpic.com/2y1zl ok"),
        f"link {URL_TOKEN} ok",
    )

    total += 1
    passed += check(
        "url www",
        replace_urls("go to www.diigo.com/~tautao today"),
        f"go to {URL_TOKEN} today",
    )

    total += 1
    passed += check(
        "url https",
        replace_urls("visit https://www.mycomicshop.com/search?TID=395031 now"),
        f"visit {URL_TOKEN} now",
    )

    total += 1
    passed += check(
        "url at start of string",
        replace_urls("http://example.com is great"),
        f"{URL_TOKEN} is great",
    )

    total += 1
    passed += check(
        "url at end of string",
        replace_urls("check out https://example.com"),
        f"check out {URL_TOKEN}",
    )

    total += 1
    passed += check(
        "url with path and fragment",
        replace_urls("see http://site.com/page#section for info"),
        f"see {URL_TOKEN} for info",
    )

    total += 1
    passed += check(
        "url with port",
        replace_urls("running on http://localhost:8080/api test"),
        f"running on {URL_TOKEN} test",
    )

    total += 1
    passed += check(
        "multiple urls",
        replace_urls("try http://a.com and https://b.com end"),
        f"try {URL_TOKEN} and {URL_TOKEN} end",
    )

    total += 1
    passed += check(
        "url case insensitive HTTP",
        replace_urls("go to HTTP://EXAMPLE.COM now"),
        f"go to {URL_TOKEN} now",
    )

    total += 1
    passed += check(
        "url case insensitive WWW",
        replace_urls("go to WWW.example.com now"),
        f"go to {URL_TOKEN} now",
    )

    total += 1
    passed += check(
        "no urls passthrough",
        replace_urls("nothing to see here"),
        "nothing to see here",
    )

    total += 1
    passed += check(
        "empty string urls",
        replace_urls(""),
        "",
    )

    # Hashtags
    total += 1
    passed += check(
        "hashtag basic",
        replace_hashtags("this is #therapyfail"),
        f"this is {HASHTAG_TOKEN}",
    )

    total += 1
    passed += check(
        "hashtag at start of string",
        replace_hashtags("#fb is trending"),
        f"{HASHTAG_TOKEN} is trending",
    )

    total += 1
    passed += check(
        "hashtag at end of string",
        replace_hashtags("check this out #AutomationAtaCost"),
        f"check this out {HASHTAG_TOKEN}",
    )

    total += 1
    passed += check(
        "hashtag with numbers",
        replace_hashtags("party like #2024"),
        f"party like {HASHTAG_TOKEN}",
    )

    total += 1
    passed += check(
        "hashtag with underscore",
        replace_hashtags("love #open_source stuff"),
        f"love {HASHTAG_TOKEN} stuff",
    )

    total += 1
    passed += check(
        "multiple hashtags",
        replace_hashtags("#happy and #sad"),
        f"{HASHTAG_TOKEN} and {HASHTAG_TOKEN}",
    )

    total += 1
    passed += check(
        "adjacent hashtags no space",
        replace_hashtags("tags: #hello#world"),
        f"tags: {HASHTAG_TOKEN}{HASHTAG_TOKEN}",
    )

    total += 1
    passed += check(
        "hashtag followed by punctuation",
        replace_hashtags("wow #cool!"),
        f"wow {HASHTAG_TOKEN}!",
    )

    total += 1
    passed += check(
        "bare # not a hashtag",
        replace_hashtags("issue # 42"),
        "issue # 42",
    )

    total += 1
    passed += check(
        "no hashtags passthrough",
        replace_hashtags("no tags here"),
        "no tags here",
    )

    total += 1
    passed += check(
        "empty string hashtags",
        replace_hashtags(""),
        "",
    )

    # Combined pipeline
    total += 1
    passed += check(
        "combined pipeline",
        preprocess_part1("hey @switchfoot check https://x.com and #fb"),
        f"hey {MENTION_TOKEN} check {URL_TOKEN} and {HASHTAG_TOKEN}",
    )

    total += 1
    passed += check(
        "combined all three adjacent",
        preprocess_part1("@user1 http://a.com #tag"),
        f"{MENTION_TOKEN} {URL_TOKEN} {HASHTAG_TOKEN}",
    )

    total += 1
    passed += check(
        "combined multiple of each",
        preprocess_part1(
            "@alice and @bob visit https://x.com and http://y.com with #hello #world"
        ),
        f"{MENTION_TOKEN} and {MENTION_TOKEN} visit {URL_TOKEN} and {URL_TOKEN} with {HASHTAG_TOKEN} {HASHTAG_TOKEN}",
    )

    total += 1
    passed += check(
        "combined url with @ and # chars",
        preprocess_part1("see https://site.com/@user#section done"),
        f"see {URL_TOKEN} done",
    )

    total += 1
    passed += check(
        "combined nothing to replace",
        preprocess_part1("just a plain sentence"),
        "just a plain sentence",
    )

    total += 1
    passed += check(
        "combined empty string",
        preprocess_part1(""),
        "",
    )

    total += 1
    passed += check(
        "combined mention before url no space issue",
        preprocess_part1("from @user99 https://link.co ok"),
        f"from {MENTION_TOKEN} {URL_TOKEN} ok",
    )

    total += 1
    passed += check(
        "combined hashtag right after url",
        preprocess_part1("link https://t.co #news end"),
        f"link {URL_TOKEN} {HASHTAG_TOKEN} end",
    )

    print(f"\nSummary: {passed}/{total} tests passed.")


def part2_tests():
    print("\nPart 2 tests: tokenization\n")

    passed = 0
    total = 0

    # BPE hand-traced training [HAND-TRACED]
    #
    # Corpus: ["aa aa aa bb"], num_merges=2
    #
    # Step 1 – GPT-2 pretokenize:
    #   "aa aa aa bb" → ['aa', ' aa', ' aa', ' bb']
    #
    # Step 2 – Word frequencies:
    #   'aa': 1,  ' aa': 2,  ' bb': 1
    #
    # Step 3 – Initial byte splits:
    #   'aa'  → ['a', 'a']
    #   ' aa' → [' ', 'a', 'a']
    #   ' bb' → [' ', 'b', 'b']
    #
    # Step 4 – Pair frequencies (iteration 1):
    #   ('a','a'): 1×1 + 1×2 = 3   (from 'aa' + ' aa')
    #   (' ','a'): 1×2 = 2          (from ' aa')
    #   (' ','b'): 1×1 = 1          (from ' bb')
    #   ('b','b'): 1×1 = 1          (from ' bb')
    #   → Best pair: ('a','a') with freq 3
    #
    # Step 5 – After merge 1  [('a','a') → 'aa']:
    #   'aa'  → ['aa']
    #   ' aa' → [' ', 'aa']
    #   ' bb' → [' ', 'b', 'b']
    #
    # Step 6 – Pair frequencies (iteration 2):
    #   (' ','aa'): 1×2 = 2   (from ' aa')
    #   (' ','b'):  1×1 = 1   (from ' bb')
    #   ('b','b'):  1×1 = 1   (from ' bb')
    #   → Best pair: (' ','aa') with freq 2
    #
    # Step 7 – After merge 2  [(' ','aa') → ' aa']:
    #   'aa'  → ['aa']
    #   ' aa' → [' aa']
    #   ' bb' → [' ', 'b', 'b']
    #
    # Final merges: [('a','a'), (' ','aa')]
    # Final vocab tokens: {' ', ' aa', 'aa', 'b'}

    bpe = BPETokenizer(num_merges=2)
    bpe.train(["aa aa aa bb"])

    total += 1
    passed += check(
        "bpe merge sequence [HAND-TRACED]",
        bpe.merges,
        [("a", "a"), (" ", "aa")],
    )

    total += 1
    passed += check(
        "bpe vocab keys [HAND-TRACED]",
        set(bpe.vocab.keys()),
        {" ", " aa", "aa", "b"},
    )

    # BPE hand-traced tokenization [HAND-TRACED]
    #
    # Tokenize "aa aa" with merges [('a','a'), (' ','aa')]:
    #   GPT-2 pretokenize → ['aa', ' aa']
    #   'aa'  → bytes ['a','a'] → merge ('a','a') → ['aa']
    #                            → merge (' ','aa') → ['aa']  (no match)
    #   ' aa' → bytes [' ','a','a'] → merge ('a','a') → [' ','aa']
    #                                → merge (' ','aa') → [' aa']
    #   Result: ['aa', ' aa']
    #
    # Tokenize "bb" (unseen during training):
    #   GPT-2 pretokenize → ['bb']
    #   'bb' → bytes ['b','b'] → no merges match
    #   Result: ['b', 'b']
    #
    # Tokenize "aa bb" (mix of seen and unseen):
    #   GPT-2 pretokenize → ['aa', ' bb']
    #   'aa'  → ['a','a'] → merge ('a','a') → ['aa']
    #   ' bb' → [' ','b','b'] → no merges match
    #   Result: ['aa', ' ', 'b', 'b']

    total += 1
    passed += check(
        "bpe tokenize seen vocab [HAND-TRACED]",
        bpe.tokenize("aa aa"),
        ["aa", " aa"],
    )

    total += 1
    passed += check(
        "bpe tokenize unseen word [HAND-TRACED]",
        bpe.tokenize("bb"),
        ["b", "b"],
    )

    total += 1
    passed += check(
        "bpe tokenize mixed [HAND-TRACED]",
        bpe.tokenize("aa bb"),
        ["aa", " ", "b", "b"],
    )

    # BPE reconstruction
    bpe_recon = BPETokenizer(num_merges=10)
    bpe_recon.train(["the cat sat on the mat"])
    for text in ["the cat", "the mat sat", "on"]:
        tokens = bpe_recon.tokenize(text)
        total += 1
        passed += check(
            f"bpe reconstruction '{text}'",
            "".join(tokens),
            text,
        )

    # BPE untrained
    bpe_raw = BPETokenizer(num_merges=0)
    bpe_raw.train(["hello"])

    total += 1
    passed += check(
        "bpe untrained returns chars",
        bpe_raw.tokenize("hi"),
        ["h", "i"],
    )

    # BPE word boundaries
    bpe_wb = BPETokenizer(num_merges=3)
    bpe_wb.train(["ab ab"])

    total += 1
    passed += check(
        "bpe word boundary merges",
        bpe_wb.merges,
        [("a", "b"), (" ", "ab")],
    )

    total += 1
    passed += check(
        "bpe word boundary tokens",
        bpe_wb.tokenize("ab ab"),
        ["ab", " ab"],
    )

    # BPE empty input
    total += 1
    passed += check(
        "bpe tokenize empty string",
        bpe.tokenize(""),
        [],
    )

    # SentencePiece BPE hand-traced training [HAND-TRACED]
    #
    # Corpus: ["aaaa"], num_merges=2
    #
    # Step 1 – No pretokenization; full_text = "aaaa"
    #   Single chunk: "aaaa" → ['a', 'a', 'a', 'a']
    #
    # Step 2 – Pair frequencies (iteration 1):
    #   Pairs (as set): {('a','a')}  → freq 1
    #   → Best (only) pair: ('a','a')
    #
    # Step 3 – After merge 1 [('a','a') → 'aa']:
    #   chunk → ['aa', 'aa']
    #
    # Step 4 – Pair frequencies (iteration 2):
    #   Pairs (as set): {('aa','aa')}  → freq 1
    #   → Best (only) pair: ('aa','aa')
    #
    # Step 5 – After merge 2 [('aa','aa') → 'aaaa']:
    #   chunk → ['aaaa']
    #
    # Final merges: [('a','a'), ('aa','aa')]
    # Final vocab: {'aaaa'}

    sp = SentencePieceBPE(num_merges=2)
    sp.train(["aaaa"])

    total += 1
    passed += check(
        "sp merge sequence [HAND-TRACED]",
        sp.merges,
        [("a", "a"), ("aa", "aa")],
    )

    total += 1
    passed += check(
        "sp vocab keys [HAND-TRACED]",
        set(sp.vocab.keys()),
        {"aaaa"},
    )

    # SentencePiece BPE hand-traced tokenization [HAND-TRACED]
    #
    # Tokenize "aaaa" with merges [('a','a'), ('aa','aa')]:
    #   bytes → ['a','a','a','a'] → merge ('a','a') → ['aa','aa']
    #   → merge ('aa','aa') → ['aaaa']
    #   Result: ['aaaa']
    #
    # Tokenize "aa" (substring):
    #   bytes → ['a','a'] → merge ('a','a') → ['aa']
    #   → merge ('aa','aa') → ['aa']  (no pair to merge)
    #   Result: ['aa']
    #
    # Tokenize "aaa" (odd length):
    #   bytes → ['a','a','a'] → merge ('a','a') → ['aa','a']  (left-to-right)
    #   → merge ('aa','aa') → ['aa','a']  (no adjacent 'aa' pair)
    #   Result: ['aa', 'a']

    total += 1
    passed += check(
        "sp tokenize full match [HAND-TRACED]",
        sp.tokenize("aaaa"),
        ["aaaa"],
    )

    total += 1
    passed += check(
        "sp tokenize partial [HAND-TRACED]",
        sp.tokenize("aa"),
        ["aa"],
    )

    total += 1
    passed += check(
        "sp tokenize odd length [HAND-TRACED]",
        sp.tokenize("aaa"),
        ["aa", "a"],
    )

    # SentencePiece crosses word boundaries
    # Unlike BPE, there is no pretokenization so the space is just another
    # byte. With enough merges the entire string collapses to one token
    # that spans the space — something BPE with pretokenization never does.
    sp_cross = SentencePieceBPE(num_merges=5)
    sp_cross.train(["ab ab"])

    total += 1
    passed += check(
        "sp merges across word boundary into single token",
        sp_cross.tokenize("ab ab"),
        ["ab ab"],
    )

    bpe_no_cross = BPETokenizer(num_merges=5)
    bpe_no_cross.train(["ab ab"])
    bpe_tokens = bpe_no_cross.tokenize("ab ab")

    total += 1
    passed += check(
        "bpe never has mid-token space (word boundary respected)",
        all(" " not in tok[1:] for tok in bpe_tokens),
        True,
    )

    # SentencePiece reconstruction
    sp_recon = SentencePieceBPE(num_merges=5)
    sp_recon.train(["the cat sat on the mat"])
    for text in ["the cat", "sat on", "mat"]:
        tokens = sp_recon.tokenize(text)
        total += 1
        passed += check(
            f"sp reconstruction '{text}'",
            "".join(tokens),
            text,
        )

    # SentencePiece untrained
    sp_raw = SentencePieceBPE(num_merges=0)
    sp_raw.train(["hello"])

    total += 1
    passed += check(
        "sp untrained returns chars",
        sp_raw.tokenize("hi"),
        ["h", "i"],
    )

    # SentencePiece empty input
    total += 1
    passed += check(
        "sp tokenize empty string",
        sp.tokenize(""),
        [],
    )

    # BPE vs SentencePiece comparison
    bpe_cmp = BPETokenizer(num_merges=3)
    bpe_cmp.train(["ab ab"])
    sp_cmp = SentencePieceBPE(num_merges=3)
    sp_cmp.train(["ab ab"])

    total += 1
    passed += check(
        "bpe vs sp produce different merge sequences",
        bpe_cmp.merges != sp_cmp.merges,
        True,
    )

    print(f"\nSummary: {passed}/{total} tests passed.")


if __name__ == "__main__":
    main()
    part2_tests()
