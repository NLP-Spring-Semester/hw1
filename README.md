# hw1
homework assignment 1 

Members

Osama Abdeljabbar  U88472172
Guhan Sambandam  U76345041

## Running Tests

```bash
python tests.py
```

## Running Part 2 (Tokenization)

```bash
uv run python part2_tokenization.py
```

## Hand-Validation of Tokenization Test Cases

Tests marked `[HAND-TRACED]` in `tests.py` were validated by manually executing each
algorithm step on paper. The full step-by-step traces are included as comments directly
above each test in the source file. Below is a summary.

### BPE (with GPT-2 pretokenization)

**Corpus:** `["aa aa aa bb"]`, `num_merges=2`

1. **GPT-2 pretokenize** splits the string respecting word boundaries. A leading
   space is attached to subsequent words:
   `"aa aa aa bb"` → `['aa', ' aa', ' aa', ' bb']`

2. **Word frequencies:** `'aa': 1, ' aa': 2, ' bb': 1`

3. **Initial byte-level splits:**
   - `'aa'`  → `['a', 'a']`
   - `' aa'` → `[' ', 'a', 'a']`
   - `' bb'` → `[' ', 'b', 'b']`

4. **Merge 1** — pair `('a','a')` has the highest frequency (3), so it is merged:
   - `'aa'`  → `['aa']`
   - `' aa'` → `[' ', 'aa']`
   - `' bb'` → `[' ', 'b', 'b']`

5. **Merge 2** — pair `(' ','aa')` now has the highest frequency (2):
   - `' aa'` → `[' aa']`

6. **Result:** merges = `[('a','a'), (' ','aa')]`, vocab = `{' ', ' aa', 'aa', 'b'}`

We then verified tokenization by replaying the merges on new input:
- `"aa aa"` → `['aa', ' aa']` (known words, fully merged)
- `"bb"` → `['b', 'b']` (unseen word, stays at byte level)
- `"aa bb"` → `['aa', ' ', 'b', 'b']` (mix of seen and unseen)

### SentencePiece BPE (no pretokenization)

**Corpus:** `["aaaa"]`, `num_merges=2`

1. **No pretokenization** — the entire text is treated as a single character
   sequence: `['a', 'a', 'a', 'a']`

2. **Merge 1** — only one unique pair exists, `('a','a')`:
   `['a', 'a', 'a', 'a']` → `['aa', 'aa']`

3. **Merge 2** — only one unique pair exists, `('aa','aa')`:
   `['aa', 'aa']` → `['aaaa']`

4. **Result:** merges = `[('a','a'), ('aa','aa')]`, vocab = `{'aaaa'}`

Tokenization verified:
- `"aaaa"` → `['aaaa']` (full match)
- `"aa"` → `['aa']` (partial — second merge has no pair to act on)
- `"aaa"` → `['aa', 'a']` (odd length — left-to-right merge leaves remainder)

### Key difference: word boundary behavior

When both tokenizers are trained on `"ab ab"`:
- **BPE** pretokenizes into `['ab', ' ab']` first, so merges can only happen
  *within* each word. The space always stays at the left edge of a token.
- **SentencePiece** sees the raw sequence `['a','b',' ','a','b']` and can merge
  the space *with adjacent characters from either side*, producing cross-word-boundary
  tokens. With enough merges, the entire string collapses to `['ab ab']`.


### The use of AI 

In completing this assignment, we made strict and responsible use of different tools as a support mechanism rather than a substitute for our own work. We primarily used AI to help with clarifying language, guiding us through our code, improving organization, and refining the presentation of ideas through graphs, especially when translating technical understanding into clear written explanations. All analysis, interpretations of results, and conclusions were developed by us based on our own implementation, observations, and critical reasoning either in person or through calls. We carefully reviewed and edited any AI-assisted content to ensure correctness, originality, and alignment with the objectives of the course. Overall, AI served as more of a supplementary tool to enhance clarity and efficiency, while the intellectual ownership and academic responsibility for the work remain entirely ours.
