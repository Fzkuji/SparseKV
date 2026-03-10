# Preliminary Attention Analysis: NIAH Token Importance

**Date**: 2026-02-22
**Models**: Qwen3-8B (baseline pretrained) vs v9 (entropy+KL fine-tuned)
**Task**: Multikey-2 style Needle-in-a-Haystack (RULER benchmark format)
**Scripts**: `scripts/show_input_and_attention.py`, `scripts/raw_attention_analysis.py`, `scripts/retrieval_head_analysis.py`, `scripts/deep_token_analysis.py`

---

## 1. Experimental Setup

### Input Sample

```
Line  0: A special magic number is hidden within the following text.     ← INSTRUCTION
         Make sure to memorize it. I will quiz you about the number afterwards.
Line  1: One of the special magic numbers for hollow-rabbit is: 5614226.  ← distractor
Line  2: One of the special magic numbers for fierce-eagle is: 3341057.
...
Line 20: One of the special magic numbers for brave-falcon is: 4829301.  ← NEEDLE
...
Line 51: One of the special magic numbers for frozen-eagle is: 1971823.
Line 52: What is the special magic number for brave-falcon                ← QUESTION
         mentioned in the provided text?
```

- **50 distractor lines** + 1 needle line + instruction + question
- **1111 tokens** total
- Needle at line 20 (tokens [425, 446))
- Question at line 52 (tokens [1095, 1111))
- Last token: `?` at position 1110

### Token Classification

| Category | Count | Percentage |
|---|---|---|
| distractor_txt | 594 | 53.5% |
| distractor_val | 350 | 31.5% |
| punctuation | 103 | 9.3% |
| instruction | 27 | 2.4% |
| question | 16 | 1.4% |
| needle_ctx | 11 | 1.0% |
| needle_value | 7 | 0.6% |
| needle_key | 3 | 0.3% |

### Needle Token Detail

```
  425: [   needle_ctx] 'One'
  426: [   needle_ctx] ' of'
  427: [   needle_ctx] ' the'
  428: [   needle_ctx] ' special'
  429: [   needle_ctx] ' magic'
  430: [   needle_ctx] ' numbers'
  431: [   needle_ctx] ' for'
  432: [   needle_key] ' brave'       ← KEY start
  433: [   needle_key] '-f'
  434: [   needle_key] 'alcon'        ← KEY end
  435: [   needle_ctx] ' is'
  436: [   needle_ctx] ':'
  437: [   needle_ctx] ' '
  438: [ needle_value] '4'            ← VALUE start
  439: [ needle_value] '8'
  440: [ needle_value] '2'
  441: [ needle_value] '9'
  442: [ needle_value] '3'
  443: [ needle_value] '0'
  444: [ needle_value] '1'            ← VALUE end
  445: [   needle_ctx] '.\n'
```

---

## 2. Last-Token Attention: Average over All Heads

The last token (pos 1110, `?`) is the generation position. Its attention distribution determines which KV entries inform the model's answer.

### Per-Line Attention (Baseline, avg over 36 layers × 32 heads)

```
  [   0-  26]   50.02%  ██████████████████████████   ← INSTRUCTION (attention sink)
  [  27-  47]    1.58%  ███
  [  48-  68]    1.07%  ██
  [  69-  89]    0.73%  █
  ...
  [ 404- 424]    0.21%                               ← distractor before needle
  [ 425- 445]    0.38%                               ← NEEDLE (indistinguishable!)
  [ 446- 466]    0.21%                               ← distractor after needle
  ...
  [ 739- 759]    2.75%  █████                        ← anomalous distractor
  ...
  [1074-1094]    1.40%  ██                           ← last distractor (position bias)
  [1095-1110]   27.26%  ███████████████              ← QUESTION
```

**Key observation**: NEEDLE line (0.38%) is completely indistinguishable from surrounding distractor lines (0.2-0.4%) in the average attention view.

### Top-10 Most Attended Tokens

```
#1  pos=0    'A'           43.2%   ← first token (attention sink)
#2  pos=1110 '?'            7.7%   ← last token (self)
#3  pos=1095 'What'         2.6%   ← question start
#4  pos=743  'magic'        2.5%   ← random distractor word
#5  pos=1105 'mentioned'    2.4%   ← question
#6  pos=1107 'the'          2.2%   ← question
#7  pos=1109 'text'         2.0%   ← question
#8  pos=1096 'is'           1.7%   ← question
#9  pos=7    'the'          1.6%   ← instruction
#10 pos=1108 'provided'     1.4%   ← question
```

Needle tokens are NOT in the top-50 of the average attention ranking.

---

## 3. Retrieval Heads: The Real Mechanism

While the average attention ignores the needle, **specific heads** attend to it strongly.

### Identified Retrieval Heads (Baseline Qwen3-8B)

| Layer | Head | Needle Attention (from last token) |
|---|---|---|
| 7 | 22 | **26.9%** |
| 13 | 14 | **18.9%** |
| 9 | 9 | 7.0% |
| 15 | 13 | 4.6% |

Only 2-4 out of 1152 (layer, head) pairs significantly attend to the needle.

### Per-Line Attention Comparison: Average vs Retrieval Heads

```
                ALL_AVG    L7H22     L13H14     L9H9     L15H13
INSTRUCTION:     50.02%    29.68%     27.08%    31.57%    56.87%
NEEDLE:           0.38%    26.90%     18.94%     7.00%     4.57%
QUESTION:        27.26%     6.37%     42.56%     5.63%     9.21%
avg distractor:  ~0.3%     ~0.5%     ~0.2%      ~1%      ~0.5%
```

**Retrieval heads give needle 27% attention vs 0.38% average — a 70× difference.**

### Which Needle Tokens Do Retrieval Heads Attend To?

**Layer 7 Head 22** (from last token):
```
  brave     0.03%
  -f        0.41%
  alcon     1.72%   ███
  is        3.03%   ██████
  :         6.84%   █████████████
  (space)   8.25%   ████████████████
  4         6.40%   ████████████
  8         0.21%
  2-9-3-0-1 ≈0%    ← remaining digits get zero attention!
```

**Layer 13 Head 14** (from last token):
```
  alcon     0.12%
  is        4.39%   ████████
  :        12.70%   █████████████████████████
  (space)   1.26%   ██
  4         0.44%
  rest ≈ 0%
```

**Finding**: Retrieval heads locate the "is: " pattern (the structural marker before the answer value). They do NOT directly attend to individual digit tokens. The digit information propagates through hidden states, not direct attention.

---

## 4. Retrieval Head Activation Dynamics (Key Finding)

### When does the retrieval head "activate"?

Tracking Layer 7 Head 22's attention to the needle line as we scan through different query positions:

**During distractor lines** (e.g., "silent-pine is: 7089806."):
```
  silent    → needle:  0.42%
  -p        → needle:  0.00%
  ine       → needle:  0.01%
  is        → needle:  0.01%
  :         → needle:  0.01%
  7089806   → needle: ~0.03%
  .\n       → needle:  0.06%
```
→ **Completely inactive. No interest in needle.**

**During question tokens** ("What is the special magic number for brave-falcon mentioned in the provided text?"):
```
  "What"        → needle:  0.02%     还没激活
  "is"          → needle:  0.07%
  "the"         → needle:  0.02%
  "special"     → needle:  0.12%
  "magic"       → needle:  0.23%
  "number"      → needle:  0.16%
  "for"         → needle:  0.68%     开始有点意识
  "brave"       → needle:  2.4%      ██ 开始激活!
  "-f"          → needle: 49.8%      ████████████████████████████████ 爆发!
  "alcon"       → needle: 44.3%      ████████████████████████████████
  "mentioned"   → needle: 78.3%      ████████████████████████████████ 峰值!
  "in"          → needle: 23.7%      ████████████████
  "the"         → needle:  0.1%      瞬间消失
  "provided"    → needle:  0.0%
  "text"        → needle:  0.1%
  "?"           → needle: 26.9%      ████████████████ 最后再看一次
```

**Layer 13 Head 14 更加极端：**
```
  "brave"       → needle: 29.5%
  "-f"          → needle: 88.4%      几乎全部 attention 都在 needle!
  "alcon"       → needle: 85.0%
  "mentioned"   → needle: 31.1%
  之后立刻回到 0%
```

### Interpretation

1. **Retrieval is triggered by query-key matching**: The head activates precisely when the question mentions the key ("brave-falcon"). Before that, it's dormant.
2. **Activation is instantaneous and transient**: It peaks at "-f" / "alcon" (the distinctive part of the key) and drops to near-zero within 1-2 tokens after the key ends.
3. **The head performs pattern matching**: Question key tokens → needle key tokens → structural marker ("is: ") → the model knows where the value is.
4. **"mentioned" gets peak attention (78.3%)**: This suggests the head consolidates retrieval after recognizing the full key match.
5. **The final `?` gets another 26.9%**: A second retrieval pass at the generation boundary.

---

## 5. Baseline vs v9 Comparison

### v9 Training Summary
- **Method**: Entropy + KL loss on attention distributions
- **Goal**: Concentrate attention patterns to make eviction easier

### What Changed

| Metric | Baseline | v9 | Change |
|---|---|---|---|
| Instruction attention (last tok) | 50.02% | 57.77% | +7.75% |
| NEEDLE attention (last tok) | 0.38% | 0.31% | -0.07% |
| Question attention (last tok) | 27.26% | 25.22% | -2.04% |
| Top token (pos=0 'A') | 43.2% | 47.7% | +4.5% |
| Entropy (layer 4) | 2.99 | 1.51 | -1.48 |
| Entropy (layer 32) | 1.84 | 1.37 | -0.47 |
| Effective support (layer 4) | 30.9 | 10.7 | -20.2 |
| Effective support (layer 35) | 10.3 | 6.3 | -4.0 |

### Retrieval Heads: Unchanged

| Head | Baseline → needle | v9 → needle |
|---|---|---|
| L7H22 | 26.9% | 31.1% |
| L13H14 | 18.9% | 17.3% |
| L9H9 | 7.0% | 6.0% |
| L15H13 | 4.6% | 4.9% |

Same heads, same ranking, nearly identical attention values.

### Per-Category Last-Token Attention Difference (v9 - baseline)

```
instruction:    +15.9%   ← attention increased, concentrated to sink
distractor_txt: -32.7%   ← decreased most
needle_ctx:     -22.3%   ← needle context also decreased
distractor_val: -10.7%
needle_key:      -9.2%   ← needle key decreased!
question:        -7.5%
needle_value:    -5.2%   ← needle value decreased!
punctuation:     -5.6%
```

### Conclusion on v9

v9 entropy training succeeded in **reducing attention entropy** (attention is more concentrated). But it concentrated attention on the **wrong targets** — sink/instruction tokens rather than task-relevant needle tokens. The retrieval heads were essentially unaffected by fine-tuning.

---

## 6. SnapKV Eviction Analysis

### SnapKV Score by Token Category (Baseline, avg over layers & heads)

| Category | Mean Score | Percentile Rank |
|---|---|---|
| question | 0.110294 | top 0% (highest) |
| instruction | 0.010228 | top 9% |
| needle_key | 0.000556 | top 17.5% |
| punctuation | 0.005710 | top 48% |
| distractor_txt | 0.004883 | top 42% |
| distractor_val | 0.006220 | top 74% |
| needle_value | 0.000172 | **top 90% (bottom 10%!)** |

**Needle VALUE tokens rank in the bottom 10%** of all tokens by SnapKV importance score. They are indistinguishable from random distractor value tokens.

### Per-Layer Needle Retention (SnapKV 50% compression, union over heads)

- Needle **key** tokens: kept in most layers (key is somewhat recognizable)
- Needle **value** tokens: kept in only ~5-7 / 36 layers
- **All** needle tokens: kept in only 4-5 / 36 layers

### Ablation: Ground Truth Token Importance (v9 model)

| Condition | Model Response |
|---|---|
| Full KV | "The special magic number for brave-falcon is: 4829..." (correct) |
| Mask needle key (3 tokens) | "The text does not contain a..." (model knows it can't find it) |
| Mask needle value (7 tokens) | "The number is missing. Let'..." (model detects value is missing) |
| Mask all distractor values (350 tokens) | Answers normally (distractors are irrelevant) |
| Mask all punctuation (103 tokens) | Answers normally |

**Both key and value tokens are essential. SnapKV would evict the value tokens.**

---

## 7. Key Takeaways

### The Fundamental Problem

```
                    SnapKV sees         Ground truth importance
needle_key:         top 17%             CRITICAL (must keep)
needle_value:       bottom 10%          CRITICAL (must keep)
distractor_val:     ~top 74%            IRRELEVANT (safe to evict)
```

SnapKV scoring (based on last-window attention) cannot distinguish needle values from distractor values because:
1. Individual digit tokens look identical in embedding space
2. The model retrieves through the KEY, not through the VALUE
3. Only a few retrieval heads attend to the needle, and they focus on the structural marker ("is: "), not the digits

### Retrieval Mechanism

- Only 2-4 out of 1152 (layer, head) pairs perform retrieval
- Retrieval heads activate **only when the query key matches** ("brave" → "-f" → "alcon")
- Activation is instantaneous (~3 tokens) and transient
- The head locates the structural pattern "is: " after the key, not the value digits directly
- Value information propagates through hidden states, not direct attention to digit tokens

### Implications for Eviction Strategy

1. **Static post-prefill eviction (SnapKV) is fundamentally flawed for retrieval tasks**: The eviction decision happens before/after the retrieval, not during it.

2. **Streaming/online eviction** aligns better with how retrieval actually works: the model doesn't know what's important until the question's key triggers retrieval.

3. **v9-style entropy training doesn't help**: Reducing attention entropy concentrates attention on sink tokens (path of least resistance), not on task-relevant tokens. The retrieval heads are not affected by this training.

4. **Per-head awareness matters**: Most heads legitimately don't need the needle. Only retrieval heads do. An eviction strategy that respects per-head differences could preserve retrieval while achieving high compression on other heads.

---

## 8. Distractor Attention Patterns

How do distractor tokens attend to earlier content? Does the model treat needle differently from distractors during intermediate processing?

**Script**: `scripts/distractor_attention.py`

### Cross-Attention Matrix (Last Token of Each Line → All Earlier Lines)

Each line's last token shows a consistent attention pattern:

| Target | Attention from a typical line |
|---|---|
| Instruction | 40-50% (attention sink, stable across all lines) |
| Immediately previous line | ~17-19% (strong recency bias) |
| 2 lines back | ~5% |
| 5 lines back | ~1-2% |
| 10+ lines back | ~0.3-0.7% |
| NEEDLE from question (line 52) | 0.38% (identical to adjacent distractors at 0.21-0.64%) |

**Every line exhibits the same pattern**: instruction (~45%) + previous line (~18%) + rapid decay. NEEDLE is completely indistinguishable from any distractor when viewed from any later position.

### Key Token Attention (Animal-Name Tokens in Each Line)

For the distinctive key token (e.g., "-river", "-eagle") in each distractor line:

```
                    → instruction   → needle line   → own line   → other distractors
Before needle:        46-73%          0.000%         15-27%         0-37%
After needle:         39-52%          0.4-4.8%       14-20%        30-45%
NEEDLE itself:        49.91%         15.83%          15.83%        18.43%
```

- Before needle: attention to needle = exactly 0% (needle hasn't appeared yet)
- After needle: attention to needle = 0.4-5%, comparable to any other distractor line
- L7H22 (retrieval head) shows sporadic spikes from some post-needle distractors (e.g., line 40: 8.09%, line 47: 13.27%), but mostly near 0%

### Retrieval Head (L7H22): Needle Attention from Every Line's Last Token

```
Lines 0-19  (before needle):    L7H22 → needle = 0.0000%     Completely dormant
Line 20     (needle itself):    L7H22 → needle = 0.3023%     avg all heads = 18.17%
Lines 21-51 (after needle):     L7H22 → needle = 0.05-5.5%   Sporadic, mostly near 0%
Line 52     (QUESTION):         L7H22 → needle = 26.9007%    Massive activation!
```

The retrieval head is completely dormant during distractor processing. It only activates when the question triggers query-key matching.

### Internal Attention Within a Line: Distractor vs Needle

Token-by-token comparison of line 10 (distractor: "swift-river") vs line 20 (NEEDLE: "brave-falcon"):

```
Line 10 (distractor):          → instr   → prev_dist  → own_line  → sink(pos0)
  "One"                         52.65%     44.30%        3.04%       45.13%
  "swift"                       54.77%     32.78%       12.45%       49.53%
  "-r" (key)                    49.90%     32.57%       17.54%       45.78%
  "3" (last digit)              47.67%     28.56%       23.76%       42.68%
  ".\n"                         49.42%     32.66%       17.92%       41.18%

Line 20 (NEEDLE):              → instr   → prev_dist  → own_line  → sink(pos0)
  "One"                         47.12%     49.89%        2.98%       41.92%
  "brave"                       49.98%     37.80%       12.23%       45.73%
  "-f" (key)                    49.91%     34.26%       15.83%       46.24%
  "1" (last digit)              45.91%     30.53%       23.56%       42.09%
  ".\n"                         46.65%     35.18%       18.17%       40.79%
```

**The patterns are virtually identical.** Same ~50% instruction, ~30% previous distractors, ~15-24% own line distribution. Needle and distractor are completely indistinguishable at the token level during forward processing.

### Key Conclusions

1. **Distractor attention is highly local**: instruction (~50%) + previous line (~18%) + own line (~15-20%), with rapid decay for distant content
2. **NEEDLE is invisible during intermediate processing**: no head, no layer, no token treats it differently from any other distractor
3. **Information propagates through hidden states, not attention**: despite zero direct attention to needle from distant positions, the model's hidden states carry enough information for retrieval when triggered
4. **Retrieval head (L7H22) is completely dormant during distractor processing**: it only activates at the question, confirming that retrieval is a late, query-triggered mechanism

---

## 9. Failure Analysis: SnapKV Eviction Breaks Copy Mechanism

**Script**: `scripts/failure_analysis.py` | **Job**: 26247 | **Date**: 2026-02-22

### 9.1 Failure Statistics

Generated 45 NIAH samples (3 distractor counts [30,50,80] × 5 needle positions [0.1,0.25,0.5,0.75,0.9] × 3 keys). 14/45 samples wrong even with full KV (excluded). Among 31 valid samples tested with SnapKV:

| Compression Ratio | Correct | Wrong | Accuracy |
|:-:|:-:|:-:|:-:|
| cr=0.3 | 26 | 5 | 83.9% |
| cr=0.5 | 19 | 12 | 61.3% |
| cr=0.7 | 6 | 25 | 19.4% |

- Middle needle positions (0.25-0.50) fail most
- In ALL 42 failures, VALUE tokens were evicted in all 36 layers
- KEY tokens evicted in 33.5/36 layers on average

### 9.2 Root Cause: Evaluation-Time vs Usage-Time Mismatch

SnapKV scores token importance using the **last W tokens' attention** (question part) at the end of prefill. But the retrieval head L7H22 at prefill time only attends to:
- Structural markers: "is:" (1.86%), space (2.10%)
- **First digit '7' only** (9.42%)
- All other digits ≈ 0%

This is because L7H22 is a **sequential copy head** that reads digits one-at-a-time during generation (see Section 10). At prefill, it hasn't started copying yet, so only the first digit gets any attention.

**GQA dilution** further weakens the signal: query head 22 shares KV head 5 with heads 20, 21, 23 (none of which are retrieval heads), diluting importance scores by 4×.

---

## 10. Generation Attention: L7H22 is a Sequential Copy Head

**Script**: `scripts/generation_attention.py` | **Job**: 26250 | **Date**: 2026-02-22
**Sample**: dist=30, pos=0.1, key=mystic-thunder, value=7156842

### 10.1 Full KV — Correct Output "7156842"

L7H22 acts as a **precise sequential pointer**: at each generation step, it attends to the exact digit that needs to be output next.

| Generated Token | L7H22 → Target Digit | Attention |
|:-:|:-:|:-:|
| (space) | VAL '7' | 9.42% |
| **7** | VAL '7' | **19.63%** |
| **1** | VAL '1' | **2.20%** |
| **5** | VAL '5' | **3.76%** |
| **6** | VAL '6' | **3.25%** |
| **8** | VAL '8' | **2.56%** |
| **4** | VAL '4' | **7.23%** |
| **2** | VAL '2' | **66.41%** |

L13H14 provides lookahead, attending to current + next digit with increasing strength (33.8% → 46.5% to value region).

### 10.2 SnapKV cr=0.5 — Wrong Output "7249183"

**Eviction (694 → 347 tokens)**:

| Token | L7H22 (KV head 5) | L13H14 (KV head 3) |
|:-:|:-:|:-:|
| VAL '7' | KEPT | KEPT |
| VAL '1' | KEPT | EVICTED |
| VAL '5' | KEPT | EVICTED |
| VAL '6' | EVICTED | EVICTED |
| VAL '8' | EVICTED | EVICTED |
| VAL '4' | EVICTED | EVICTED |
| VAL '2' | EVICTED | EVICTED |

**After eviction, L7H22 loses its copy targets and redirects attention to "for" token**:

| Generated Token (wrong) | L7H22 → "for" (pos 95) | L7H22 → all needle kept |
|:-:|:-:|:-:|
| (space) | **94.53%** | 0.47% |
| **7** | **61.72%** | 4.68% |
| **2** (should be 1) | **89.45%** | 0.34% |
| **4** (should be 5) | **90.23%** | 0.01% |
| **9** (should be 6) | **97.66%** | 0.19% |
| **1** (should be 8) | **76.95%** | 1.55% |
| **8** (should be 4) | **90.63%** | 0.18% |
| **3** (should be 2) | **98.05%** | 0.03% |

The copy head can't find its targets, collapses to an anchor token ("for"), and the model hallucinates random digits.

### 10.3 Key Insight

The model reads multi-digit values **one digit at a time** during generation, like a sequential pointer. SnapKV evaluates importance at prefill time when the pointer hasn't started moving yet, so it only sees the first digit as "important." This is a fundamental **evaluation-time vs usage-time mismatch**.

---

## 11. Prefill Attention: Period "." as Span Summary Token

**Script**: `scripts/prefill_attention.py` | **Job**: 26251 | **Date**: 2026-02-22

### 11.1 Period "." Broadly Attends to All Value Digits

The period "." at the end of the needle sentence (pos 113), during prefill, attends broadly to **all 7 value digits** in many heads:

| Layer, Head (KV head) | '7' | '1' | '5' | '6' | '8' | '4' | '2' | Total |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| L5 H8 (KV2) | 5.4% | 6.9% | 8.9% | 12.9% | 12.9% | 10.7% | 8.9% | **66.6%** |
| L6 H30 (KV7) | 15.4% | 11.7% | 11.0% | 8.0% | 13.7% | 10.3% | 11.3% | **81.4%** |
| L6 H20 (KV5) | 0.9% | 1.2% | 0.5% | 0.5% | 1.6% | 13.2% | 80.9% | **98.7%** |
| L7 H25 (KV6) | 8.5% | 4.0% | 6.2% | 4.4% | 6.6% | 14.9% | 26.2% | **70.9%** |
| L8 H22 (KV5) | 3.3% | 5.0% | 4.2% | 1.7% | 4.9% | 11.7% | 22.5% | **53.2%** |
| L11 H11 (KV2) | 9.4% | 15.5% | 13.8% | 3.9% | 16.6% | 3.1% | 8.3% | **70.7%** |
| L12 H29 (KV7) | 3.9% | 3.1% | 3.8% | 3.9% | 5.8% | 27.9% | 21.7% | **70.1%** |

L5 H8 and L6 H30 are especially notable: nearly **uniform attention across all 7 digits**.

### 11.2 This is a General Pattern (Not Needle-Specific)

Distractor lines' periods also attend to their own digits:

| Line | Digits | L7 avg total | L9 avg total | L13 avg total | L15 avg total |
|:-:|:-:|:-:|:-:|:-:|:-:|
| Distractor 2 | 3414798 | 13.2% | 5.7% | 5.0% | 6.6% |
| Distractor 3 | 2615384 | 9.4% | 4.3% | 1.7% | 4.5% |
| **Target** | **7156842** | **10.9%** | **5.1%** | **1.0%** | **4.4%** |

The period is a **universal sentence-summary token** — it naturally aggregates the preceding content.

### 11.3 Implication for KV Eviction

SnapKV only uses the last W tokens (question part) to score importance, missing the structural signal from "." tokens. If the "." token's attention were incorporated into scoring, **all value digits would be identified as important**, preventing the destructive eviction that breaks the copy mechanism.

This suggests a potential improvement: use **structural marker tokens** (periods, newlines, etc.) as auxiliary importance signals alongside the standard last-window scoring.

---

## 12. Method Comparison: SnapKV vs CriticalKV vs ExpectedAttention vs Knorm

**Job 26253** | Script: `method_comparison.py` | cr=0.5, dist=30, pos=0.1

通过 monkey-patch `ScorerPress.compress` 记录每层每 KV head 的保留/驱逐决策。

### 12.1 Representative Case: Value Digit Retention

Target value = "7156842" (7 digits), positions 106-112, seq_len=702

| Method | L7H22 (KV5) | L13H14 (KV3) | L9H9 (KV2) | L15H13 (KV3) |
|:--|:--|:--|:--|:--|
| **SnapKV** | kept '7','1','5' (3/7) | kept '7' (1/7) | kept '7','1','4','2' (4/7) | kept '7','4','2' (3/7) |
| **CriticalKV** | kept '7','1','5' (3/7) | kept '7' (1/7) | kept '7','1','5','4','2' (5/7) | kept '7','4','2' (3/7) |
| **ExpectedAttn** | kept '7','8','2' (3/7) | kept '7','5','6','2' (4/7) | kept '7','4','2' (3/7) | kept '2' (1/7) |
| **Knorm** | kept '1','5','6','8','4' (5/7) | kept '7','1','8' (3/7) | kept '7','1','5','6','4' (5/7) | kept **all 7** (7/7) |

### 12.2 Key Findings

1. **CriticalKV ≈ SnapKV**: `(scores + ε) × value_norm` 无法拯救 score=0 的 token。L7H22 和 L13H14 完全相同。
2. **ExpectedAttention 选择不同但同样丢失**：通过 query 统计 + RoPE 预测未来 attention，保留了不同的 digits（如 '8'），但仍丢失 4/7。
3. **Knorm 表现最好**：基于 key 向量 L2 范数（`-keys.norm(dim=-1)`），完全不依赖 attention。L15H13 保留全部 7 个 digits。但 L7H22 丢失第一个 digit '7'。
4. **所有方法都有缺陷**：没有方法能在所有 retrieval head 上同时保留全部 value digits。

### 12.3 Batch Test

由于 Qwen3 的 thinking mode（输出 `<think>...`），即使 full KV 也无法在前 20 字符内匹配 target value，导致所有 batch test cases 被跳过。Representative case 的 eviction log 仍然有效。

---

## 13. Attention Ray Tracing: Bidirectional Importance Propagation

**Job 26254** | Script: `attention_raytracing.py` | cr=0.5, dist=30, pos=0.1

核心思路：不仅看 question 直接关注什么（SnapKV），而是沿 attention chain 双向传播 importance：
- **Forward**: `boost_fwd[j] = Σ_i importance[i] × attn[i, j]` （重要 token 关注了 j）
- **Backward**: `boost_bwd[j] = Σ_i importance[i] × attn[j, i]` （j 关注了重要 token）

### 13.1 L7 层全部 KV Head 对比

| KV Head | SnapKV keeps | Ray Tracing keeps | Improvement |
|:--|:--|:--|:--|
| KV0 | '7','2' (2/7) | **all 7** (7/7) | 749× |
| KV1 | '2' (1/7) | **all 7** (7/7) | 7,089× |
| KV2 | '5','4','2' (3/7) | **all 7** (7/7) | 7,689× |
| KV3 | none (0/7) | '7','1','5','6','8' (5/7) | 19,314× |
| KV4 | none (0/7) | '5','6' (2/7) | 42,683× |
| **KV5 (H22)** | **'7','2' (2/7)** | **all 7 (7/7)** | **741×** |
| KV6 | '7' (1/7) | '7','1','5','6' (4/7) | 3,117× |
| KV7 | '2' (1/7) | '7','1','5' (3/7) | 19,894× |

KV0, KV1, KV2, KV5 达到 7/7 完美保留。其余 head 也显著改善。

### 13.2 Multi-hop 传播（2 bounces）

L7 KV5 (包含 H22):
- 0-hop (SnapKV): total score = 0.434, keeps 2/7
- 1-hop: total = 320.8, keeps 7/7
- 2-hop: total = 981.5, keeps 7/7
- Combined: total = 406.2, **keeps 7/7**

### 13.3 L7H22 单头分析

只用 H22 的 attention（不做 GQA group 平均）：
- SnapKV H22 keeps: '7','1','2' (3/7)
- **Ray tracing H22 keeps: all 7 (7/7)**

1-hop 后 value digits 的 score 从 0.6285 → 427.9（680× increase）。

### 13.4 Chain Tracing

L7 H22 的直接 attention chain:
- question → digit '7': 0.49% （只看到第一个 digit）
- question → ':': 1.41%
- '.' → value digits: 几乎为 0（H22 不是 "." 头）

但跨 head 互补：
- **H25 (KV6)**: '.' → all value digits = 70.9%（"." 是 span summary 头）
- **H18 (KV4)**: '.' → all value digits = 32.7%

Ray tracing 的关键：在同一 KV group 内，不同 query head 携带互补信息。H22 从 question 找到第一个 digit，H25 通过 "." 找到所有 digits。Bidirectional propagation 把这些信号合并到 KV head 级别。

### 13.5 Implications

1. **SnapKV 的根本问题不是算法错误，而是信息不足**：单纯看 question 对 key 的 attention 无法发现间接重要的 token。
2. **Ray tracing 可行性**：只需要 prefill attention + 1-2 hop 传播即可显著改善。计算开销：在已有 attention matrix 上做矩阵乘法，远小于重新 forward。
3. **Per-head 决策是关键**：不同 KV head 需要保留不同的 token。全局 top-k 会稀释 retrieval head 的信号。
4. **Bidirectional 很重要**：Forward（important → j）捕获「被关注者传递重要性」，Backward（j → important）捕获「关注者继承重要性」，两者缺一不可。

---

## 14. KVzip Pipeline Test (Job 26259)

使用正确的 kvpress pipeline 模式测试 KVzip（prefill 在 context manager 内，generation 在外部）。

### 测试配置
- 模型: Qwen3-8B (bfloat16)
- 任务: Needle-in-a-haystack (30 distractors, target at 10% position)
- Target: mystic-thunder → 7156842
- Context length: 697 tokens
- Compression ratio: 0.5

### 关键发现：KVzip "Fake Compression"
KVzip 的 `compress_post()` 不裁剪 cache tensor，而是把 evict 的位置存在 `module.masked_key_indices`。在 decoding 时通过 `attention_patch.py` 把这些位置的 key 替换成 fake key (使 `exp(<q,k>) ≈ 0`)。Cache tensor 大小不变，但 attention 计算时被 mask 的 token 不参与。

### 结果

| 方法 | 压缩方式 | Cache 长度 | 回答 |
|------|---------|-----------|------|
| Full KV | 无 | 697 | CORRECT — 7156842 |
| SnapKV cr=0.5 | 真裁剪 | 348 | WRONG — 7147456 |
| Knorm cr=0.5 | 真裁剪 | 348 | 值对但 key 错 ("mythical-thunder") |
| KVzip layerwise cr=0.5 | Fake mask | 697 | CORRECT — 7156842 |
| KVzip non-uniform cr=0.5 | Fake mask | 697 | CORRECT — 7156842 |

### KVzip Score 分析 (cr=0.5)

KVzip context reconstruction 打分，各 retrieval head 的 value digit 保留情况：

| Head | 保留 | Evicted | 保留率 |
|------|------|---------|--------|
| L7 KV5 (H22) | 7,6,8,4,2 | 1,5 | 5/7 |
| L13 KV3 (H14) | 1,5,6,8,4,2 | 7 | 6/7 |
| L9 KV2 (H9) | 5,6,8,4,2 | 7,1 | 5/7 |
| L15 KV3 (H13) | 2 | 7,1,5,6,8,4 | 1/7 |

不同 head 互相补充：每个 head 丢失的 digit 在其他 head 中被保留。

### Score 详情

**L7 KV5 (H22):**
- Value digit scores: [0.632812, 0.263672, 0.230469, 0.609375, 0.660156, 0.609375, 0.835938]
- Ranks: [176/697, 357/697, 378/697, 187/697, 158/697, 188/697, 65/697]

**L13 KV3 (H14):**
- Value digit scores: [0.703125, 0.843750, 0.937500, 1.000000, 0.984375, 0.980469, 0.914062]
- Ranks: [390/697, 327/697, 240/697, 9/697, 165/697, 168/697, 267/697]

**L9 KV2 (H9):**
- Value digit scores: [0.147461, 0.198242, 0.439453, 0.953125, 0.941406, 0.726562, 0.832031]
- Ranks: [379/697, 353/697, 250/697, 42/697, 47/697, 156/697, 114/697]

**L15 KV3 (H13):**
- Value digit scores: [0.041260, 0.015015, 0.008057, 0.005188, 0.002594, 0.025635, 0.188477]
- Ranks: [373/697, 506/697, 564/697, 598/697, 645/697, 440/697, 170/697]

### 结论
KVzip 的 context reconstruction scoring 优于 SnapKV 的 attention-based scoring，但代价是 2-3 倍额外计算开销。

---

## Raw Data Files

- Server logs: `/home/zichuanfu2/logs/show_attn_26241.txt` (baseline), `/home/zichuanfu2/logs/show_v9_26242.txt` (v9)
- Raw attention analysis: `/home/zichuanfu2/logs/raw_attn_26240.txt`
- Retrieval head analysis: `/home/zichuanfu2/logs/retrieval_head_26243.txt`
- Deep token analysis (SnapKV): `/home/zichuanfu2/logs/deep_token_26239.txt`
- Distractor attention analysis: `/home/zichuanfu2/logs/dist_attn_26244.txt`
- Failure analysis: `/home/zichuanfu2/logs/fail_analysis_26247.txt`
- Generation attention: `/home/zichuanfu2/logs/gen_attn_26250.txt`
- Prefill attention (period analysis): `/home/zichuanfu2/logs/prefill_attn_26251.txt`
- Structural token analysis: `/home/zichuanfu2/logs/struct_attn_26252.txt`
- Method comparison: `/home/zichuanfu2/logs/method_cmp_26253.txt`
- Ray tracing: `/home/zichuanfu2/logs/raytracing_26254.txt`
- Scripts: `~/SparseKV/scripts/{show_input_and_attention,raw_attention_analysis,retrieval_head_analysis,deep_token_analysis,distractor_attention,failure_analysis,generation_attention,prefill_attention,structural_token_analysis,method_comparison,attention_raytracing}.py`

---

## 15. 3-Hop Attention Tracing with Inverse Fan-In Weighting

**Jobs**: 26320-26329 | **Scripts**: `threehop_tracing.py`, `threehop_niah_only.py` | **Date**: 2026-02-24
**Model**: Qwen3-8B | **Task**: NIAH (30/100 distractors, needle at 50%)

### 15.1 Method Design

基于 Section 13 Ray Tracing 的发现，设计了 3-hop query-aware attention tracing：

```
Step1: Question tokens 的 OUTGOING attention → 找到 key tokens
Step2: 谁关注了 step1 高分 token（INCOMING, weighted by inverse fan-in）→ 找到 period 等桥接 token
Step3: Step2 高分 token 的 OUTGOING attention → 传导到 value tokens
```

**关键创新 — Inverse Fan-In Weighting**:
- `fan_in[j] = Σ_i attn[i,j]`（column sum，token j 被多少人关注）
- `inv_fanin = 1 / (fan_in + ε)`
- 解决 Sink Pollution：sink token (pos 0 "A": fan_in=46.05) 被所有人关注，如果在 step2 中找"谁关注了 sink"，会把所有 token 都拉进来。Inverse fan-in 让 sink 的传播权重极低，只有被少数 token 特异性关注的 key 才传播 importance。

### 15.2 实现细节与关键条件

**必须条件 1: Max-over-groups（非 Mean）**

Qwen3-8B: 32 query heads / 8 KV heads = 4 groups。Retrieval 行为只出现在 1-2 个 query group 中。
- Mean: 信号被 3 个不相关 group 稀释 → FAIL
- Max (amax over group dim): 保留最强 group 的信号 → OK

**必须条件 2: Exact Fan-In（非 K-norm 近似）**

K-norm 近似 (`fan_in ≈ ||K_j|| × sqrt(pos)`) 严重低估 sink 的 fan-in。Sink 的高 fan-in 来自 learned attention bias，不是 K 向量大小。
- K-norm proxy: FAIL（100d cr=0.7 失败）
- Exact column sum (2-pass chunked): OK

**内存优化**: Chunked 计算，O(chunk_size × L) per iteration，避免 O(L²) 全量 attention 矩阵。

### 15.3 NIAH 结果 (v3b, Job 26329)

| 配置 | Full KV | cr=0.5 global | cr=0.7 global | cr=0.9 global |
|------|---------|-------------|-------------|-------------|
| 30d | OK | OK | **OK** | FAIL |
| 100d | OK | OK | **OK** | FAIL |

- cr=0.7 全部通过（30d 和 100d 均 OK）
- cr=0.9 全部失败
- Per-head allocation 在 cr=0.7 也失败（信噪比不够）

### 15.4 版本对比

| 版本 | Fan-in | Group 聚合 | 30d cr=0.7 | 100d cr=0.7 |
|------|--------|-----------|-----------|------------|
| v0 (full attn) | Exact | Max | OK | OK |
| v1 (K-norm) | K-norm proxy | Max | OK | FAIL |
| v2 (K-norm+mean) | K-norm proxy | Mean | FAIL | FAIL |
| v3 (chunked+K-norm) | K-norm proxy | Max | OK | FAIL |
| **v3b (chunked+exact)** | **Exact** | **Max** | **OK** | **OK** |

---

## 16. Token Ablation — Minimal Token Set for NIAH

**Jobs**: 26331, 26332 | **Scripts**: `token_ablation.py`, `token_ablation_v2.py` | **Date**: 2026-02-24

### 16.1 Needle Token 分类

NIAH needle: "One of the special magic numbers for **mystic-thunder** is: **7156842**."

| 类别 | Positions | Tokens |
|------|-----------|--------|
| Prefix | 343-349 | "One of the special magic numbers for" |
| **Key** | 350-353 | "myst", "ic", "-th", "under" |
| Sep | 354-356 | "is", ":", " " |
| **Value** | 357-363 | "7", "1", "5", "6", "8", "4", "2" |
| Period | 364 | ".\n" |

Base tokens（总是保留）: sink[0:4] + recent[662:694] = 36 tokens

### 16.2 Ablation v1 — 单组件测试 (Job 26331)

| Test | 保留内容 | Tokens | 结果 |
|------|---------|--------|------|
| 0 | Full KV | 694 | **OK** |
| 1 | base only (sink+recent) | 36 | FAIL |
| 2 | base + full needle | 58 | **OK** |
| 3 | base + key + value | 47 | **OK** |
| **4** | **base + key only** | 40 | **FAIL** |
| **5** | **base + value only** | 43 | **OK** |
| 6 | base + key + value + period | 48 | **OK** |
| 10 | base + prefix + key (NO value) | 47 | **FAIL** |
| **11** | **base + value + period (NO key)** | 44 | **OK** |
| 12 | base + period only | 37 | FAIL |
| 13 | everything EXCEPT target needle | 672 | FAIL |
| **15** | **needle only in L20-35** | — | **OK** |
| **16** | **needle only in L0-19** | — | **FAIL** |

### 16.3 Ablation v2 — 消歧测试 (Job 26332)

v1 Test 5 "value only OK" 是否因为只有一个数字可见？

| Test | 描述 | 结果 |
|------|------|------|
| A1 | target value + 5 nearest distractor values (NO keys) | **OK** |
| A2 | target value + ALL 30 distractor values (NO keys) | **OK** |
| **A3** | **WRONG value only (distractor value)** | **输出了错误数字** |
| B6 | target value only + 5 nearest distractor FULL needles | **OK** |
| **C3** | **target key ONLY + ALL full distractors (no target value)** | **FAIL** |

A3 证实模型会输出唯一可见数字。但 A1/A2 证明即使 31 个数字同时可见且无 key，模型照样选对 target value。

### 16.4 核心发现

1. **Value tokens 是必须的，Key tokens 不是**: Test 4 (key only) FAIL，Test 5 (value only) OK。C3 (key only + all distractors) FAIL。
2. **模型不通过 key→value 链式查找**: 31 个数字可见时，无 key 也能选对（A2 OK），说明 question 中的 key 在 prefill 时已经将关联信息编码到了 value tokens 的 KV cache representation 中。
3. **Period 不重要**: Test 3 (no period) 和 Test 6 (with period) 结果相同。
4. **深层（L20-35）是关键**: Test 15 OK, Test 16 FAIL。Retrieval 主要发生在深层。
5. **最小充分集**: sink + recent + value tokens = 43 tokens（94% 压缩率）。

---

## 17. CR=0.9 Failure Analysis

**Jobs**: 26330, 26333 | **Scripts**: `threehop_cr09_analysis.py`, `cr09_deep_analysis.py` | **Date**: 2026-02-24

### 17.1 Score 分布问题

Needle tokens 在 3-hop combined score 中只处于 60-72% 百分位：

| Token 类别 | Score 百分位 | cr=0.7 保留率 | cr=0.9 保留率 |
|-----------|------------|-------------|-------------|
| Key | 65-72% | 23-36% | 9-13% |
| Value | 60-68% | 17-27% | 7-11% |
| Period | 59.6% | 18.4% | 5.2% |

cr=0.9 阈值在 P90=1.2113，而 needle tokens 平均得分只有 0.26-0.44，远低于阈值。

### 17.2 Question 直接关注 Value 的层级分布

| Layer | Question→Key | Question→Value | 说明 |
|-------|-------------|---------------|------|
| L0-L16 | <0.02 | <0.025 | 两者都弱 |
| **L20** | 0.099 | **0.111** | Value 最高！Question 直接关注 value |
| L24 | **0.250** | 0.019 | 切换到只看 key |
| L28 | **0.248** | 0.000 | Value 完全为零 |
| L32 | **0.273** | 0.000 | 只看 key |

**模型检索分两阶段**:
- L20: 同时关注 key 和 value（value 甚至更高）
- L24+: 只关注 key，value 注意力归零

### 17.3 谁关注 Value Tokens？

Top-20 中主要是 value 自己（self-attention）和少量 question token（"myst" 等）。**Period token 对 value 的直接注意力不强。**

### 17.4 Per-Layer Heatmap

cr=0.9 时 value tokens 在各层的保留率：
- **L0-L16**: 完全被删（0% 保留）
- **L18-L20**: 开始出现，但很稀疏（1-2/8 heads）
- **L24-L34**: 每个 value token 平均 2-3/8 heads 保留

Value 得分最高的 (layer, head) 对：L28 H7, L21 H0, L19 H5

### 17.5 可视化

保存于 `retention_vis/` 目录：

| 文件 | 内容 |
|------|------|
| `token_retention_cr70.png` | 全局 token 保留率 (cr=0.7) |
| `token_retention_cr90.png` | 全局 token 保留率 (cr=0.9) |
| `needle_cr07_vs_cr09.png` | Needle 区域 cr=0.7 vs cr=0.9 对比 |
| `perlayer_heatmap_cr70.png` | 每层 needle token 保留热力图 (cr=0.7) |
| `perlayer_heatmap_cr90.png` | 每层 needle token 保留热力图 (cr=0.9) |
| `score_distribution.png` | Score 分布 + needle token 位置 |
| `diff_cr07_vs_cr09.png` | cr=0.7 保留但 cr=0.9 丢失的 token |

---

## 18. Cross-Layer Information Flow — 单层 3-Hop 的根本限制

### 18.1 问题

3-hop 方法在每层独立计算三步。但真实的信息传递链是跨层的：

```
Layer ~5-8:   Period "." → Value digits (Section 11: 66-99% attention)
Layer ~18-20: Question → Value (direct, 0.111 attention)
Layer ~24-32: Question → Key (retrieval heads, 0.25-0.27 attention)
```

当我们在 L32 算 3-hop 时：
- Step1 在 L32 找到 key ✓（question→key = 0.273）
- Step2 在 L32 找到 period ✓（period 关注 key）
- Step3 看 period 在 **L32** 关注什么 → **period→value 在 L32 很弱** ✗

Period→value 的强信号在 L5-L8 层（Section 11 数据），但 step1 在这些浅层也很弱。三步假设的链条分布在不同层，但算法在每层独立计算，链条断裂。

### 18.2 Value Tokens 在 Prefill 时的真实情况

Value digits 在 prefill 阶段没有被强烈关注，因为模型**在生成时才逐个读取 digits**（Section 10: L7H22 sequential copy head）。Prefill 时唯一直接关注 value 的机会在 L20（attention 0.111），之后完全消失。

这意味着 3-hop 能给 value 的最大信号来自 L20 的 step1 直接注意力，而不是通过 period 的间接传导。但 L20 的信号在 36 层 normalize+sum 后被淹没。

### 18.3 根本矛盾

| 需求 | 方法能力 |
|------|---------|
| 保留 value digits（生成时需要） | 3-hop 对 value 的 score 只有 60-68% 百分位 |
| 在 prefill 时识别 value 重要性 | 模型自身在 prefill 时也不特别关注 value |
| 跨层传播 key→period→value | 单层 3-hop 无法跨层传播 |

核心矛盾：**Value digits 的重要性只在生成时才真正体现（sequential copy），但 eviction 决策必须在 prefill 时做出。** 这是所有 post-prefill eviction 方法（SnapKV, CriticalKV, 3-hop）共同面临的 evaluation-time vs usage-time mismatch（Section 10.3）。

---

## Raw Data Files (continued)

- 3-hop tracing: `/home/zichuanfu2/SparseKV/logs/threehop_niah_26329.out`
- CR=0.9 analysis: `/home/zichuanfu2/SparseKV/logs/cr09_analysis_26330.out`
- Token ablation v1: `/home/zichuanfu2/SparseKV/logs/token_ablation_26331.out`
- Token ablation v2: `/home/zichuanfu2/SparseKV/logs/token_ablation_v2_26332.out`
- Deep analysis: `/home/zichuanfu2/SparseKV/logs/cr09_deep_26333.out`
- Retention visualization: `/home/zichuanfu2/SparseKV/logs/retention_vis/`
- Scripts: `~/SparseKV/scripts/{threehop_niah_only,threehop_cr09_analysis,token_ablation,token_ablation_v2,cr09_deep_analysis,visualize_retention}.py`

---

## 19. Question Attention Per-Head Deep Dive

**Jobs**: 26338-26339 | **Scripts**: `vis_question_attn_v2.py`, `step1_topk_analysis.py` | **Date**: 2026-02-24
**Model**: Qwen3-8B | **Task**: NIAH (30 distractors, needle at 50%, key=mystic-thunder)
**Context**: 694 tokens, needle at pos 343-364, key tokens at pos 350-353 ("myst","ic","-th","under")

### 19.1 Retrieval Heads 识别（Per Query Head）

V1 可视化用了 max-over-all-KV-heads，导致所有 category 的 attention 都被拉到 ~1.0。V2 改为逐 query head 分析：

| Layer | Query Head | →Key Attn | Key/Dist Ratio | 角色 |
|-------|-----------|-----------|----------------|------|
| L24 | QH23 | **0.9900** | 48,728× | Retrieval head |
| L28 | QH3 | **0.9954** | extreme | Retrieval head |
| L28 | QH23 | **0.4929** | 4,270× | Retrieval head |
| L32 | QH8 | 0.3660 | 5,163× | Retrieval head |
| L20 | QH4 | 0.2428 | — | Weak retrieval |

其他 ~27 个 query head：~20 个做 question self-attention，~5 个做 sink attention，几乎没有关注 distractor。

### 19.2 Step1 Top-K Token Analysis (Per KV Head, L24)

Step1 = max over groups & question tokens（和 3-hop 一样）。

| KV Head | Top tokens 主要类别 | Needle Key Rank |
|---------|-------------------|-----------------|
| KV0 | Question tokens ("myst","special","magic") + "about","within" | ~456 |
| KV1 | Question + "oon","hidden","quiz","number","text" | ~562 |
| KV2 | "number","it", question, "magic","special"（模板词） | ~500+ |
| KV3 | 模板词 + question | ~384 |
| KV4 | 类似 | ~582 |
| **KV5** | **sink("A") + "the" + "is" → needle value rank 5, key rank 18** | **~18** |
| KV6 | 其他 distractor 词 | 高 |
| KV7 | 类似 | 高 |

**结论**：只有 KV5 是 retrieval head（QH23 在 KV5 group 中），其他 head 关注的是 question 自身 token、模板/指令词、高频通用词。

---

## 20. Step1 Fixed-CR Eviction Test

**Job**: 26342 | **Script**: `step1_eviction_test.py` | **Date**: 2026-02-24

### 20.1 方法

用 Step1 scores（question outgoing attention, max over groups & q tokens）对每层每 head 独立选 top-K，zeroing out 被 evict 的 KV entries，然后生成。

### 20.2 结果

| 方法 | CR=0.3 | CR=0.5 | CR=0.7 | CR=0.8 | CR=0.9 | CR=0.95 |
|------|--------|--------|--------|--------|--------|---------|
| Step1 | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ |
| SnapKV | — | — | ✗ | — | — | — |
| Oracle (force-keep needle) | — | — | — | — | ✓ | ✗ |

- Step1 在 CR=0.5 还能工作，CR=0.7 开始失败
- Oracle 证明：保留 needle 22 tokens + top-K 其余 token，cr=0.9 仍然正确
- cr=0.95 即使 Oracle 也失败（非 needle 上下文不够）

---

## 21. Needle Token Rank Analysis

**Job**: 26346 | **Script**: `step1_needle_rank.py` | **Date**: 2026-02-24

### 21.1 Key Tokens Rank（Best Head 视角）

"在最好的那个 head 里，needle key tokens 排第几？"

| Layer | First Token Rank | Max CR | Key Tokens Worst Rank | Max CR |
|-------|-----------------|--------|----------------------|--------|
| L20 | 14 | **98.0%** | 97 | **86.0%** |
| L24 | 98 | 85.9% | 145 | 79.1% |
| L28 | 131 | 81.1% | 82 | **88.2%** |
| L32 | 125 | 82.0% | 87 | 87.5% |

### 21.2 Per-Head 独立判断（现实场景）

每个 head 独立根据自己的 attention 决定保留什么。要保住全部 4 个 key tokens：

| Layer | KV0 | KV1 | KV2 | KV3 | KV4 | KV5 | KV6 | KV7 | 瓶颈 |
|-------|-----|-----|-----|-----|-----|-----|-----|-----|------|
| L20 | 45% | 86% | 23% | 56% | 49% | 39% | 13% | 12% | **12%** (KV7) |
| L24 | 61% | 70% | 52% | 45% | 16% | 65% | 27% | 71% | **16%** (KV4) |
| L28 | 61% | 16% | 63% | 74% | 43% | 42% | 28% | 8% | **8%** (KV7) |
| L32 | 0% | 0% | 0% | 74% | 26% | 12% | 54% | 2% | **0%** (KV0/1/2) |

**结论**：Per-head 独立判断时，大部分 head 根本不关注 needle，瓶颈 head 在很低的 CR 就会 evict needle。

---

## 22. Per-Head Cross-Layer Union Eviction

**Job**: 26347 | **Script**: `step1_perhead_union.py` | **Date**: 2026-02-24

### 22.1 方法

1. 每个 KV head 每层选 top-K tokens（K = keep_ratio × ctx_len）
2. 跨 36 层 union → 该 head 的 keep set
3. 同一个 keep set 应用到所有层

### 22.2 Union 膨胀

| 每层 Top-K | Union Size/Head | Actual CR | Key Coverage (all heads) |
|-----------|----------------|-----------|------------------------|
| top-13 (2%) | ~195 | 72.5% | 部分 |
| top-34 (5%) | ~370 | 46.7% | 6/8 heads 全保 |
| top-69 (10%) | ~528 | 23.8% | **8/8 全保** |
| top-104 (15%) | ~612 | 11.8% | 8/8 全保 |
| top-138 (20%) | ~652 | 6.0% | 8/8 全保 |
| top-208 (30%) | ~688 | 0.9% | 8/8 全保 |

36 层 union 导致严重膨胀：每层 top-5% 只有 34 个 token，union 后变成 ~370（53%）。

### 22.3 Eviction 正确性

| 每层 Top-K | Actual CR | 正确? |
|-----------|-----------|-------|
| top-13 (2%) | 72.5% | ✗ |
| top-34 (5%) | 46.7% | ✗ |
| top-69 (10%) | 23.8% | ✗ |
| **top-104 (15%)** | **11.8%** | **✓** |
| top-138 (20%) | 6.0% | ✓ |
| top-208 (30%) | 0.9% | ✓ |

需要每层 top-15%（actual CR 仅 11.8%）才能保证正确。

### 22.4 关键发现

1. **Cross-layer union 导致实际压缩率远低于名义值**：每层 top-5% → actual CR 仅 47%，每层 top-30% → actual CR 仅 0.9%
2. **原因**：36 层每层的 top-K 不太重叠（不同层关注不同 token），union 快速覆盖全部 token
3. **Needle key 在 top-10% 时所有 head 全保**，但答案仍错 → 光保住 key 不够，还需要足够的上下文 token

---

## 23. Adaptive Eviction 尝试（失败记录）

**Jobs**: 26343-26345 | **Date**: 2026-02-24

### 23.1 Attention Mass Threshold (Job 26343)

每个 question token 保留 top-X% attention mass 的 positions。

**全部失败**（threshold 0.5-0.99，CR 32-48%）。原因：每层用自己的 attention 决定保留什么，大多数层不关注 needle。

### 23.2 Iterative Softmax (Job 26344)

迭代 softmax 试图 sharpen 分布。**反而 smoothing 到 uniform**。原因：softmax(probabilities) 对 [0,1] 范围的输入是压缩而非放大。iter=3 时全保（694/694），iter=5+ 时只剩 8 个（sink+recent）。

### 23.3 Power Sharpening (Job 26345)

修正为 `p^α / Σp^α`（正确的 sharpening）。Per-layer 和 global 方式均全部失败。根本问题相同：per-layer attention 无法跨层传播 needle 信息。

---

## Raw Data Files (continued)

- Question attention v2: `/home/zichuanfu2/SparseKV/logs/vis_question_attn_v2_26338.out`
- Step1 top-k analysis: `/home/zichuanfu2/SparseKV/logs/step1_topk_26339.out`
- Step1 eviction test: `/home/zichuanfu2/SparseKV/logs/step1_evict_26342.out`
- Adaptive threshold: `/home/zichuanfu2/SparseKV/logs/step1_adaptive_26343.out`
- Iterative softmax: `/home/zichuanfu2/SparseKV/logs/step1_itersoft_26344.out`
- Power sharpening: `/home/zichuanfu2/SparseKV/logs/step1_power_26345.out`
- Needle rank analysis: `/home/zichuanfu2/SparseKV/logs/needle_rank_26346.out`
- Per-head union: `/home/zichuanfu2/SparseKV/logs/perhead_union_26347.out`
- Scripts: `~/SparseKV/scripts/{vis_question_attn_v2,step1_topk_analysis,step1_eviction_test,step1_adaptive_test,step1_iterative_softmax,step1_power_sharpen,step1_needle_rank,step1_perhead_union}.py`

---

## 24. Two-Hop Method: Period-Based Anchor + Value Recovery

**Jobs**: 26348-26383 | **Date**: 2026-02-24
**Model**: Qwen3-8B | **Task**: NIAH (150 distractors, 50 samples, needle at random positions)

### 24.1 核心思路

放弃在 prefill 时直接识别 value tokens（Section 18 证明这是根本矛盾），转而利用 **period "." 作为 sentence-level anchor**：

```
Hop1: Question differential attention → 找到 target sentence 的 period "."
Hop2: Period 的 outgoing attention → 找到该 sentence 的 value tokens
```

**依据**：Section 11 证明 period 是 universal sentence-summary token，在 L5-8 层对 value digits 有 66-99% 的聚合 attention。

### 24.2 Differential Attention Scoring

对每个 period position j，计算 question tokens 相比非 question tokens 的 **差异注意力**：

```python
q_mean[j] = mean(attn[question_tokens → j])        # question 对 j 的平均注意力
baseline[j] = sum(attn[non_question_tokens → j]) / count(non_question_visible_from_j)  # causal mask aware
diff[j] = q_mean[j] - baseline[j]                   # 差异分数
```

关键：baseline 必须考虑 causal mask（每个 period 只能被它后面的 token 看到），否则位置在后面的 period 会因为 visible token 少而 baseline 偏大。

### 24.3 Baseline 50-Sample Results (SnapKV, Expected, KVzip, KVzip+)

**Job**: 26387 | **Script**: `baseline_50samples.py`

| Method | CR50% | CR70% | CR80% | CR85% | CR90% | CR95% |
|--------|-------|-------|-------|-------|-------|-------|
| SnapKV | 16% | 6% | 4% | 2% | 2% | 0% |
| Expected | 26% | 6% | 0% | 0% | 0% | 0% |
| KVzip | 100% | 100% | 98% | 98% | 94% | 56% |
| **KVzip+** | **100%** | **100%** | **100%** | **98%** | **98%** | **66%** |

KVzip+ (non-uniform) 在 CR90% 仍有 98% 准确率，是最强 baseline。

### 24.4 Two-Hop Narrow Hop2 Layer Range

**Job**: 26383 | **Script**: `twohop_highcr.py`

All-layer Hop2 (L0-12) 导致 CR 只有 78%。缩窄 Hop2 层范围恢复 CR：

| Config | Accuracy | Mean CR |
|--------|----------|---------|
| k10 L0-12 n10 | 96% | 78% |
| k10 L3-8 n10 | **96%** | **85%** |
| k10 L5-8 n10 | 94% | 88% |
| k5 L3-8 n10 | 74% | 91% |

L3-8 是最佳 Hop2 层范围（Section 11 数据支持：period→value 主要在 L5-8）。

### 24.5 Per-Layer Adaptive Hop2（失败）

**Job**: 26386 | **Script**: `twohop_adaptive.py`

每层独立选择 keep set（而非所有层共享同一 keep set）。

**结果：0% 准确率**。大多数层（L12-35）对 value tokens 的覆盖率接近 0。一旦某层丢失了 value 的 KV entry，信息链断裂。**所有层必须共享同一 keep set。**

---

## 25. Hop1 Optimization: Union, Intersection, Aggregate

**Job**: 26388 | **Script**: `twohop_hop1opt.py` | **Date**: 2026-02-24

### 25.1 问题

Hop1 用 L19-23 的 5 层 union（每层 top-10 periods），结果约 20 个 anchor periods，CR 只到 85%。能否减少 anchors？

### 25.2 策略比较

| Strategy | Accuracy | Target Hit Rate | Mean Anchors | Mean CR |
|----------|----------|-----------------|-------------|---------|
| **union_k10** (L19-23) | **96%** | **98%** | 20.2 | 84.8% |
| intersect_k10 (L19-23) | 0% | 0% | 0.0 | — |
| aggregate_top5 | 22% | 24% | 5.0 | 93.7% |
| aggregate_top3 | 8% | 10% | 3.0 | 95.6% |
| L21_top5 | 72% | 74% | 5.0 | 93.5% |
| L21_top3 | 56% | 62% | 3.0 | 95.5% |
| L21_top1 | 16% | 18% | 1.0 | 97.1% |

Intersection 过于严格（0 个 overlap）；aggregate 和 single-layer 方法精度不够。

---

## 26. Weighted Hop1 Allocation: Breaking 90% CR

**Job**: 26390 | **Script**: `twohop_weighted_hop1.py` | **Date**: 2026-02-24

### 26.1 核心思路

分析发现 L21 是最重要的 Hop1 层。给 L21 单层分配更多 budget（k=15 或 k=20），不再依赖多层 union。

### 26.2 结果

| Config | Accuracy | Mean CR | Target Hit | Failures |
|--------|----------|---------|------------|----------|
| **L21_only_k15** | **96%** | **90.4%** | **96%** | S11 (modelFAIL), S45 (Hop1 MISS) |
| **L21_only_k20** | **98%** | **87.2%** | **98%** | S11 only |
| L21_15 + L20_5 | 96% | 89.6% | 96% | S11, S45 |
| uniform_k10 (baseline) | 96% | 84.8% | 98% | S11 |

**L21_only_k15 首次突破 90% CR！** 96% accuracy @ 90.4% CR。

### 26.3 失败分析

**Job**: 26389 | **Script**: `twohop_failure_analysis.py`

仅 2 个失败样本：
- **S11**: fullFAIL（模型本身也无法正确回答）— 不是方法的错
- **S45**: Hop1 MISS — target period 在 L21 diff_max 排名第 16（超出 k=15 budget）

L21 diff_max rank 分布（50 samples）：
```
≤1: 24%   ≤3: 58%   ≤5: 74%   ≤10: 96%   ≤15: 98%
```

---

## 27. Ratio Scoring: Normalizing Positional Bias (Key Discovery)

**Job**: 26391 | **Script**: `twohop_deep_s45.py` | **Date**: 2026-02-24

### 27.1 问题

S45 中 L21 diff_max 把 target period 排在第 16 位。能否找到更好的 scoring 方式？

### 27.2 发现：Ratio vs Diff

```python
diff[j] = q_mean[j] - baseline[j]    # 原方法
ratio[j] = q_mean[j] / baseline[j]   # 新方法
```

| Metric | diff_max | ratio_max |
|--------|----------|-----------|
| S45 target rank | **16** | **2** |
| 50-sample mean rank | 4.1 | **1.4** |
| Top-5 coverage | 37/50 | **49/50** |
| Top-10 coverage | 48/50 | **50/50** |

### 27.3 为什么 Ratio 更好

diff (减法) 对 **绝对注意力量高** 的 token 有偏好。序列前端的 period（pos ~50）和 instruction 后的 period 因为 positional bias / sink 效应，baseline 和 q_mean 都很高，diff 也大，容易"抢占"排名。

ratio (除法) 归一化了 baseline 幅度：
- 前端 token: q_mean=0.01, baseline=0.008 → diff=0.002 (高), ratio=1.25 (低)
- Target period: q_mean=0.005, baseline=0.001 → diff=0.004, ratio=5.0 (高)

Ratio 衡量的是 "question 对这个 token 的注意力**相对于背景**有多异常"，而非绝对值。

### 27.4 S45 详细分析

diff_max 排名 1-5 的 period 都是位于序列前端的结构性位置（pos=50, 3184, instruction text），它们的高 diff 来自 positional bias 而非 question-specific retrieval signal。

ratio_max 正确识别 target period 为 rank 2，因为 target period 相对于 baseline 的 attention uplift 最显著。

---

## 28. Ratio Scoring Full Eviction Test

**Job**: 26392 (RUNNING) | **Script**: `twohop_ratio.py` | **Date**: 2026-02-24

### 28.1 测试配置

- Hop1: L21 only, ratio_max / diff_max / diff_mean × k=3,5,7,10
- Hop2: L3-8, n=10
- 50 samples, 150 distractors

### 28.2 完整结果 (Job 26392 COMPLETED)

| Strategy | Accuracy | Mean CR | Min CR | Val Coverage | Target Hit |
|----------|----------|---------|--------|-------------|------------|
| **ratio_max_k3** | **94% (47/50)** | **96.8%** | 96.6% | 6.7/7 | 48/50 |
| **ratio_max_k5** | **96% (48/50)** | **95.5%** | 95.1% | 6.9/7 | 49/50 |
| **ratio_max_k7** | **96% (48/50)** | **94.3%** | 93.7% | 6.9/7 | 49/50 |
| **ratio_max_k10** | **98% (49/50)** | **92.5%** | 91.6% | 7.0/7 | 50/50 |
| diff_max_k3 | 56% (28/50) | 97.1% | 96.7% | 4.1/7 | 29/50 |
| diff_max_k5 | 72% (36/50) | 96.2% | 95.4% | 5.2/7 | 37/50 |
| diff_max_k7 | 76% (38/50) | 95.4% | 94.3% | 5.5/7 | 39/50 |
| diff_max_k10 | 94% (47/50) | 93.7% | 92.9% | 6.7/7 | 48/50 |
| diff_mean_k3 | 38% (19/50) | 97.5% | 96.9% | 2.7/7 | 19/50 |
| diff_mean_k5 | 66% (33/50) | 96.8% | 96.1% | 4.8/7 | 34/50 |
| diff_mean_k7 | 94% (47/50) | 95.5% | 95.1% | 6.7/7 | 48/50 |
| diff_mean_k10 | 96% (48/50) | 93.4% | 92.8% | 7.0/7 | 50/50 |

### 28.3 Key Takeaways

1. **ratio_max_k10: 98% @ 92.5% CR** — 超越 KVzip (94% @ 90%) 且压缩率更高！
2. **ratio_max_k5: 96% @ 95.5% CR** — 在 95% CR 下仍有 96% 准确率，远超所有 baseline
3. **ratio_max 全面碾压 diff_max**：同 k 值下，ratio_max 准确率高 2-38 个百分点
4. S45 在 ratio_max 下成功 Hit（diff_max rank 16 → ratio_max 进入 top-3）
5. diff_max_k10 反而掉到 47/50（S35 和 S45 Miss），进一步证实 ratio 优势

### 28.4 Comparison with All Baselines

| Method | CR | Accuracy | Notes |
|--------|-----|----------|-------|
| SnapKV | 90% | 2% | 完全失败 |
| ExpectedAttn | 80% | 0% | 完全失败 |
| KVzip | 90% | 94% | 强 baseline |
| KVzip+ | 90% | 98% | 最强 baseline |
| diff_max L21 k15 | 90% | 96% | 之前最好 |
| **ratio_max k10** | **92.5%** | **98%** | **匹配 KVzip+，CR 更高** |
| **ratio_max k5** | **95.5%** | **96%** | **远超 KVzip，接近 KVzip+** |
| **ratio_max k3** | **96.8%** | **94%** | **极端压缩下仍可用** |

---

## 29. 200-Sample Large-Scale Evaluation

**Jobs**: 26393 (NIAH ratio), 26394 (NIAH KVzip+) | **Date**: 2026-02-25

### 29.1 NIAH: Two-Hop ratio_max (200 samples, 150 distractors)

**Job 26393** | **Script**: `twohop_ratio_200.py`

Full KV baseline: 193/200 correct (7 model failures: S011, S071(?), S131, S140, S146, S150, S159, S191)

**排除 fullFAIL (193 valid samples):**

| Strategy | Accuracy | Mean CR | Min CR | Target Hit |
|----------|----------|---------|--------|------------|
| **ratio_k10** | **191/193 (99.0%)** | **92.4%** | 91.3% | 192/193 |
| ratio_k15 | 191/193 (99.0%) | 89.5% | 88.0% | 193/193 |
| **ratio_k7** | **189/193 (97.9%)** | **94.2%** | 93.3% | 191/193 |
| **ratio_k5** | **187/193 (96.9%)** | **95.4%** | 94.9% | 190/193 |
| ratio_k3 | 179/193 (92.7%) | 96.8% | 96.4% | 183/193 |

**含全部 200 samples:**

| Strategy | Accuracy | Mean CR |
|----------|----------|---------|
| ratio_k10 | 191/200 (95.5%) | 92.4% |
| ratio_k7 | 189/200 (94.5%) | 94.2% |
| ratio_k5 | 187/200 (93.5%) | 95.4% |
| ratio_k3 | 179/200 (89.5%) | 96.8% |

**方法失败仅 2 例**（ratio_k10）：
- S071: Hop1 MISS (hit=False, val=0/7)
- S120: Hit but wrong (hit=True, val=7/7, 可能 generation error)

### 29.2 NIAH: KVzip+ (200 samples, 150 distractors)

**Job 26394** | **Script**: `kvzip_200.py`

| CR | Accuracy (200 samples) |
|----|----------------------|
| 80% | **200/200 (100.0%)** |
| 85% | **197/200 (98.5%)** |
| 90% | 193/200 (96.5%) |
| 92.5% | 186/200 (93.0%) |
| 95% | 122/200 (61.0%) |
| 97% | 30/200 (15.0%) |

### 29.3 NIAH Head-to-Head Comparison (200 samples)

| Method | CR ~90% | CR ~92.5% | CR ~95% | CR ~97% |
|--------|---------|-----------|---------|---------|
| KVzip+ | **96.5%** @ 90% | 93.0% @ 92.5% | 61.0% @ 95% | 15.0% @ 97% |
| **ratio_k10** | — | **95.5%** @ 92.4% | — | — |
| **ratio_k7** | — | — | **94.5%** @ 94.2% | — |
| **ratio_k5** | — | — | **93.5%** @ 95.4% | — |
| **ratio_k3** | — | — | — | **89.5%** @ 96.8% |

**结论**：
1. CR<90% 区间 KVzip+ 略优（100% @ 80% vs 我们不测这个区间）
2. **CR≥92.5% 我们全面领先**：ratio_k10 95.5% vs KVzip+ 93.0%（+2.5%）
3. **CR≥95% 差距巨大**：ratio_k5 93.5% vs KVzip+ 61.0%（+32.5%!）
4. **CR≥97% 彻底碾压**：ratio_k3 89.5% vs KVzip+ 15.0%（+74.5%!）
5. KVzip+ 在高 CR 时急剧崩溃，我们的方法降级优雅

---

## 30. FactQA: Knowledge Base QA Task (Generalization Test)

**Jobs**: 26397 (ratio), 26399 (KVzip+, running) | **Date**: 2026-02-25

### 30.1 Task Design

150 fictional location entries, each with:
```
{Location} was founded in {year} and has a population of {population}.
The current mayor is {mayor} and the elevation is {elevation} meters.
```

Question types:
- "What is the population of {loc}?" (40%, 7-digit answer)
- "In what year was {loc} founded?" (40%, 4-digit answer)
- "What is the elevation of {loc} in meters?" (20%, 3-digit answer)

Tests generalization: different text format, multiple value types, natural language QA.

### 30.2 FactQA: Two-Hop ratio_max (200 samples)

**Job 26397** | **Script**: `factqa_ratio.py`

Full KV baseline: 197/200 correct (3 model failures)

**排除 fullFAIL (197 valid samples):**

| Strategy | Accuracy | Mean CR | Target Hit |
|----------|----------|---------|------------|
| **ratio_k15** | **182/197 (92.4%)** | **92.6%** | 179/197 |
| **ratio_k10** | **178/197 (90.4%)** | **94.8%** | 175/197 |
| ratio_k7 | 172/197 (87.3%) | 96.2% | 173/197 |
| ratio_k5 | 163/197 (82.7%) | 97.1% | 160/197 |
| ratio_k3 | 139/197 (70.6%) | 97.8% | 126/197 |

**按问题类型（排除 fullFAIL）：**

| Type | ratio_k10 Acc | ratio_k15 Acc | Samples |
|------|-------------|-------------|---------|
| population (7-digit) | ~高 | ~高 | — |
| year (4-digit) | 29/61 (47.5%) | 31/61 (50.8%) | 61 |
| elevation (3-digit) | 3/3 (100%) | 3/3 (100%) | 3 |

**观察**：
- Population 问题表现较好（类似 NIAH）
- **Year 问题准确率明显低（~50%）** — 4-digit year 可能有其他匹配问题
- Elevation 样本太少无法判断
- 方法在不同格式下仍然有效（90.4% @ 94.8% CR），但不如 NIAH（99.0% @ 92.4% CR）

### 30.3 FactQA: KVzip+ (200 samples)

**Job 26399** — RUNNING，待完成后更新

---

## Raw Data Files (continued)

- Two-hop high CR: `/home/zichuanfu2/SparseKV/logs/highcr_26383.out`
- Two-hop adaptive (failed): `/home/zichuanfu2/SparseKV/logs/adaptive_26386.out`
- Baseline 50 samples: `/home/zichuanfu2/SparseKV/logs/baseline50_26387.out`
- Hop1 optimization: `/home/zichuanfu2/SparseKV/logs/hop1opt_26388.out`
- Failure analysis: `/home/zichuanfu2/SparseKV/logs/failure_26389.out`
- Weighted Hop1: `/home/zichuanfu2/SparseKV/logs/weighted_26390.out`
- Deep S45 analysis: `/home/zichuanfu2/SparseKV/logs/deep_s45_26391.out`
- Ratio scoring 50s: `/home/zichuanfu2/SparseKV/logs/ratio_26392.out`
- NIAH ratio 200s: `/home/zichuanfu2/SparseKV/logs/ratio200_26393.out`
- NIAH KVzip+ 200s: `/home/zichuanfu2/SparseKV/logs/kvzip200_26394.out`
- FactQA ratio 200s: `/home/zichuanfu2/SparseKV/logs/factqa_ratio_26397.out`
- FactQA KVzip+ 200s: `/home/zichuanfu2/SparseKV/logs/factqa_kvzip_26399.out` (running)
- Scripts: `~/SparseKV/scripts/{twohop_ratio_200,kvzip_200,factqa_ratio,factqa_kvzip}.py`
