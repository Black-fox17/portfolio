# Tokenization and Encoding with Byte Pair Encoding

Tokenization is an integral part of Natural Language Processing (NLP), which is the frontier of how our modern LLMs came to be. You see, computers don't understand language like humans do, but they have the ability to understand numbers and can calculate much faster than humans can.

Tokenization defines what the symbols are. Encoding defines how those symbols are mapped to numbers. They are both part of the preprocessing steps that allows us to represent language much more efficiently and better so computers can understand.

## Why Tokenization Matters

Since I said computers understand numbers far better than humans can, a question might pop up in your mind: **why can't we just represent all alphabets by a number?** Then encode our text corresponding to that number for instance, all occurrences of "a" will be 1 and "z" will be 26. 

Okay, easy there, genius! Well, this in simple terms is what **encoding** in NLP means: representing tokens (or as you might refer to them, "words" well, it's not just words, we'll get there in a moment) as numbers. Let's say `a = 1`, `b = 2`, and so on. We can represent "hello" in numbers as `[8, 5, 12, 12, 15]`. 
**This is character-level tokenization**, **It is not what modern llms use**

The computer will surely understand this, but for every solution we come up with, there's always room for an efficiency problem. This means the number of digits we use to represent a word is the same length as our word itself—"hello" is 5 characters long, same with `[8, 5, 12, 12, 15]`. This is not efficient and leads to wastage of space and computational resources because, my friend, for your LLM or any language-based model to be trained on a neural network, it passes through operations and representations.

So, our genius, you came up with another idea: **representing each word by a single number**. Now this does seem efficient! At least now we can represent a word like "hello" as `1` and "world" as `3`, so our encoding representation will be `[1, 3]` for "helloworld". Don't see this as a typo yes, "helloworld" is different from "hello world". The space `" "` in terms of NLP is also a token, so we have to assign a number for it too. "hello world" might be something like `[1, 2, 3]`.

This does seem practical—we just have to assign an ID to every word in the dictionary. Wait... there are over **500,000 English words** alone! Well, we also have to consider morphology: "run" can be "running", "runner", etc. What about other languages? Oh, there are emojis, symbols too, slang too! This will make our representation balloon into millions of tokens. Well, oh genius, this doesn't seem effective to me.

This is what gave birth to some of the current tokenization and encoding techniques today. Most of them started out as **compression techniques** to fully represent language.

![BPE merge](https://curator-production.s3.us.cloud-object-storage.appdomain.cloud/uploads/course-v1:IBMSkillsNetwork+GPXX0A7BEN+v1.jpg)

## Overview of Tokenization Techniques

Like I said earlier, what you invented is in fact a tokenization technique! Separating words into letters is a form of tokenization called **Character-level tokenization**, which we both know has efficiency issues. But it does have its strength: it doesn't encounter the **OOV (Out-of-Vocabulary)** problem, which is simply not having a number to represent a token. Since at the foundation level all languages have characters, and we can simply assign these characters a number, this means all words can easily be encoded.

**Word-level tokenization**, which I discussed earlier, does offer a better way of compression compared to character-level tokenization, but fails with OOV due to the fact we can't account for all possible words.

This is what modern tokenization techniques called **subword-level tokenization** try to fix. We have a bunch: **Unigram**, **WordPiece**, **BPE**, etc. I won't be going much into the details of other techniques aside from BPE, but they all strive to achieve maximum compression and effective representation.

![](https://substackcdn.com/image/fetch/$s_!tXNg!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F0196ea0d-95a4-4ad2-bf9f-15262a630fd8_1654x1036.webp)
Remember I mentioned something regarding tokens not being words alone? They can be characters or subwords. For instance, tokens can be `["hel", "eat", "!", "@", "g", "ing", "ed", "1", "2"]` these are all tokens, and this is what the subword-level techniques try to achieve. It allows for better representation of words. Like "hello" can become `["he", "ll", "o"]`. This depends on what kind of tokens we have: "he" as a unique ID, same as "ll" and "o", and for all occurrences of "he" we represent them with its ID.

**Voilà!** We significantly improve compression while retaining generalization. We can fully represent all words, even ones as complex as **"Donaudampfschiffahrtselektrizitätenhauptbetriebswerkbauunterbeamtengesellschaft"** (this is a German compound word). Now with less than **60,000 tokens**, we can represent all languages' words. Sounds like magic, right? I know.

## Why Byte-Level BPE (GPT-Style)

Byte Pair Encoding was actually introduced by **Gage in 1994** as a form of compression algorithm. It is quite a simple technique, yet the most effective technique to grasp. It is the frontier behind the tokenization concept used for the preprocessing of data for training modern Large Language Models or in other major NLP tasks.

### The Optimization Problem

There are two major optimization problems in BPE:

**Optimal Merge Sequence (OMS) Problem:**  
Given a string $s$ and an integer $k > 0$, find a merge sequence $R$ of length $k$ with maximal utility for $s$ (or equivalently, of minimal compressed length). We denote this optimal utility as $\text{OPT}_m(s, k)$.

**Optimal Pair Encoding (OPE) Problem:**  
Given a string $s$ and an integer $k > 0$, find a partial merge sequence $R^*$ of length $k$ with maximal utility for $s$. We denote this optimal utility as $\text{OPT}(s, k)$.

**Byte-Pair Encoding (BPE) Algorithm:**  
BPE solves both the OPE and OMS problems as follows. Starting with the input string $s$, it performs $k$ locally optimal full merge steps, always choosing a pair whose replacement maximizes compression utility.

Formally, for input $(s, k)$, we output $R = (R_1, \ldots, R_k)$, where $R_i = \text{replace}_{a_i b_i \rightarrow c_i}$. Denoting $s^{(0)} = s$, and $s^{(i)} = R_i(s^{(i-1)})$ for $i \in [k]$, each $c_i$ is a new symbol (i.e., not occurring in $s^{(j)}$ with $j < i$), and for $i = 1, \ldots, k$, the pair $a_i b_i$ is chosen so that $|R_i(s^{(i-1)})|$ is minimal.



$$\text{BPE}(s, k) \leq \text{OPT}_m(s, k) \leq \text{OPT}(s, k)$$

**BPE is greedy it optimizes local compression, not global optimality.**

### The Byte-Level Breakthrough

![](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTtg0NF_l2rJLYgnL0zYPty2xilCf-M2QA-2g&s)

Although we won't dive too much into optimization problems here, rather how it relates to LLMs, BPE solves the OOV problem and also handles emojis, accents, and rare scripts which was a major issue concerning tokenization—simply by borrowing the concept that **"every string is representable as bytes"**. This simple line changes everything. How? you might ask. A Chinese character can be represented as bytes!

Before byte-level encoding, tokenizers worked on characters. This sounds fine for English, but the world has:

- Over **140,000 different Unicode characters** (including emojis like 🚀 and complex scripts like Chinese or Hindi)

If your "base" vocabulary has to include every single possible Unicode character just to avoid the `<UNK>` (unknown token), your vocabulary is already "exploded" before you even start merging pairs!

By falling back to **bytes (UTF-8 encoding)**, we simplify the entire universe of text into just **256 possible base values**.

### How This Changes Everything

**The Universal Base:**  
Instead of having a "base" vocabulary of 140,000+ characters, the "base" is just 256. This is tiny and manageable.

**No More `<UNK>`:**  
Since every possible piece of digital text (from a simple 'a' to a complex 🇨🇳 flag) is ultimately just a sequence of bytes between 0 and 255, the model can always represent the input.

**Efficiency:**  
BPE then starts merging these bytes. Common English letters might merge quickly, while a complex emoji might stay as a sequence of 3 or 4 byte-tokens. A Chinese character can be represented by 2 or more bytes.

## BPE Algorithm Implementation

### Step 1: Pre-tokenization

The first part of building a BPE tokenizer is **pre-tokenization** using regex. As it was done in the GPT-2 BPE implementation, they used:

```regex
r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
```

This ensures handling of contractions, punctuation, and spaces. A simple "Hello world!" text is separated into:

```
"Hello world!" → ["Hello", " world", "!"]
```

### Step 2: Byte-to-Unicode Mapping

As we mentioned about the Unicode byte representation with 256 base bytes, we can now represent any characters in our words—symbols or Chinese characters can be represented by two or more bytes. This allows the conversion into characters constrained by the 256 base bytes, and with UTF-8 we can easily reverse the character back to its original symbol. This is such an important task in our BPE algorithm.

Here is code that shows the byte-to-unicode representation:

```python
def bytes_to_unicode(self):
    """
    Returns list of utf-8 bytes and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a significant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1)) + \
         list(range(ord("¡"), ord("¬")+1)) + \
         list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))
```

The base alphabet consists of **256** byte-derived Unicode symbols, not raw bytes. (excluding special tokens), which are all the possible UTF-8 encoded characters.

### Step 3: Training Loop

The training loop is such a simple thing to grasp—it is a **popularity contest algorithm** which favors the most frequently occurring pairs.

**During tokenization, merges are applied strictly by rank, ensuring deterministic segmentation.**

If we have a training corpus:

```python
corpus = [
    {"text": "low lower"},
    {"text": "low lowest"}
]
```

We extract words: `low`, `lower`, `low`, `lowest`

Calculate each word's frequency:

```python
{
    ('l', 'o', 'w'): 2,
    ('l', 'o', 'w', 'e', 'r'): 1,
    ('l', 'o', 'w', 'e', 's', 't'): 1
}
```

For each word, we extract pairs:

**Pairs per word:**

```
('l', 'o', 'w') × 2
    → (l, o), (o, w) → each count += 2

('l', 'o', 'w', 'e', 'r')
    → (l, o), (o, w), (w, e), (e, r)

('l', 'o', 'w', 'e', 's', 't')
    → (l, o), (o, w), (w, e), (e, s), (s, t)
```

**Total pair counts:**

```
(l, o): 4
(o, w): 4
(w, e): 2
(e, r): 1
(e, s): 1
(s, t): 1
```

Then we select the best pair (let's say `(l, o)`) and add it to our vocab as `"lo"`. We also create a **merge dictionary** to track the merges, so during the next iteration the word "low" will be represented as `('lo', 'w')`.

This is done until we reach our `vocab_size`. GPT-2's tokenizer has over **50,000 vocab size** trained over large datasets. This ensures it can capture better relationships, and the frequency is more favorable.

**Now you just built the BPE training algorithm!**

### Step 4: Tokenization Process

The tokenization process follows the same process as our training algorithm, but now we have a vocab built on ranking that we can query. When we compute pairs, we can query the `vocab_rank` to check if the pair exists, then we proceed to merge the pair if it exists.

### Step 5: Encoding

Encoding is just using our `byte_encoder` to encode the text with UTF-8 and decode with the 256 base bytes. This prevents OOV. Then we tokenize, and we can query the tokenized tokens in our vocab and get the encoded IDs. This is built **strictly** by ranks.

Please notice the use of "strictly"—it really matters!

### Step 6: Decoding

Decoding follows the same process, just in reverse order.

**And yeah, my friend, you just built a BPE Tokenizer!**

## Limitations

It has some limitations, especially with the frequency computation. Like I said, it is a popularity contest algorithm, but the sky is wide you can invent better algorithms, my friend!

## Why LLMs Can't Spell

As an addition, the major reason behind the famous question LLMs couldn't answer is: **"How many R's in strawberry?"** Tokenization contributes to this issue, but it’s not the sole cause.

If you check the word "strawberry" on [TikTokenizer](https://tiktokenizer.vercel.app/?model=gpt2), you can see the token "strawberry" is broken down into three different tokens: `[301, 1831, 8396]` → `["st", "raw", "berry"]`. This is how this is fed into the LLM, so the LLM doesn't see characters but tokens!


You can check out the full code at [BPETokenizer](https://github.com/Black-fox17/llm_scratch/blob/main/tokenization/tokenizer.py)
## Key Takeaway

**Tokenization defines symbols.**  
**Embeddings define meaning.**

We will look at embeddings in the next blog post.

---

**Thanks for reading!**

If you have any questions or suggestions, you can reach out to me via email at  
[Mail](mailto:ayeleru1234@gmail.com)