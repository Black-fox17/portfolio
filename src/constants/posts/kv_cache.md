# KV Cache and Long-Context Transformers: How Autoregressive Models Scale at Inference

Training a Large Language Model (LLM) is one challenge; inference is another. Inference primarily deals with text generation, a process that has undergone significant modification for performance optimization and efficiency since the inception of the Transformer architecture.

## The Evolution of Sequence Modeling

Early sequence modeling relied on Recurrent Neural Networks (RNNs) and variants like LSTM (Long Short-Term Memory). While text generation isn't new, these architectures suffered from a major drawback: effective handling of long contexts. Previous models struggled to learn patterns and relationships over long sequences, leading to rapid context degradation.

The Transformer architecture, introduced by the Vaswani team, solved this with the mechanism of **Attention**. It allowed for better scaling and context alignment over longer sequences processed in parallel. However, this came with its own costs. As context length grows, the standard attention mechanism computing attention scores for a token and all its preceding tokens ($QK^T / \sqrt{d} \cdot V$) introduces a significant bottleneck.

### The Inference Bottleneck

In a naive implementation, generating each new token requires recomputing attention scores for the entire sequence history. This results in $O(T^2)$ time complexity, where $T$ is the sequence length. While this quadratic cost is acceptable during training (where parallelization is easier), it becomes a massive hurdle during autoregressive inference, where tokens are generated one by one.

If you send a message to an LLM and it responds in milliseconds, **KV Cache** (Key Value Cache) is likely the reason for that lightning fast generation.

## What is KV Cache?

The Key Value Cache (KV Cache) aims to fix the $O(T^2)$ computational bottleneck by storing the computed Key and Value vectors of previous tokens. Instead of recomputing them at every step, the model retrieves them from memory. This reduces the redundant matrix multiplications effectively. KV caching eliminates recomputation of past keys and values, reducing total inference cost across a sequence from $O(T^2)$ to $O(T)$. Each decoding step still attends over the growing cache, but avoids redundant projection work.

### Visualizing the Cache Data

Below is a visualization from huggingface showing the KV Cache data structure.

Here is a comparison of the two methods. The first video shows the without KV Cache, and the second shows with KV Cache.
![without KV Cache](/blog/cache/without.mp4)

![With KV Cache](/blog/cache/with.mp4)

Caching the key-value pairs does not degrade the semantics of the generation if done correctly. We are not modifying the training process, but rather optimizing the inference stage.

## Mechanics of Inference

In autoregressive modeling, we use a causal mask to prevent past tokens from "seeing" future tokens.
1.  **Without KV Cache**: At step $T$, we compute the Key and Value matrices for all tokens $1$ to $T$.
2.  **With KV Cache**: At step $T$, we only compute the Key and Value for the *new* token at $T$. We then concatenate this with the cached Keys and Values from steps $1$ to $T-1$.

## Case Study: Performance Comparison

Let's look at a qualitative comparison of text generation.

**Scenario**: Generating a simple story.

**Note**: When using greedy decoding with a fixed seed, outputs with and without KV cache are identical. Any observed differences below are due to stochastic sampling, not caching.

**Without KV Cache:**
> Once upon a time, there was a little boy named Tim. Tim loved pumpkins. One day, he saw a big slide and he looked up in the sky. He was very sad. Tim did not like the big wave and he had a bad ending. Tim was very happy and said he was sorry.

**With KV Cache:**
> Once upon a time, there was a little boy named Tim. Tim loved pumpkins. One day, Tim wanted to go to the park with his mom. Tim told his mom about the spaceship, so he asked his mom if he could go outside.

As you can see, the semantics are preserved (differences in output are due to sampling temperature or seeds, not the caching mechanism itself). The real difference lies in the speed.

### Latency Comparison

| Tokens | No KV (s) | KV Cache (s) | Speedup |
| :--- | :--- | :--- | :--- |
| 200 | 5.1641 | 1.2052 | **4.28x** |
| 500 | 23.4041 | 3.3650 | **6.96x** |
| 1000 | 62.3627 | 7.1104 | **8.77x** |
| 2000 | 140.6186 | 14.3266 | **9.82x** |

As the token length grows, the speedup increases significantly, validating the efficiency of KV Caching.

## Challenges in Implementation

### Positional Encodings
Modernizing transformer architecture often involves changing the positional encoding technique from absolute (sinusoidal or learned) to relative techniques like **RoPE** (Rotary Positional Embeddings) or **ALiBi**. When using RoPE, the query and key vectors are rotated to encode position. When caching, one must ensure that cached keys are correctly handled relative to the current position. RoPE itself is compatible with KV caching. Most failures arise from positional misalignment during cache warm up or incorrect handling of incremental position indices, not from caching rotated keys per se

### Memory Constraints
There is a major trade off with speed: **Memory**. As context grows, memory usage for the KV Cache scales linearly with context length (and batch size), eventually leading to VRAM exhaustion.

## Advanced Optimizations

To handle memory bottlenecks, researchers have introduced techniques like **Sparse Attention** and **Sliding Window Attention**.

### Sliding Window Attention
Instead of attending to all preceding tokens, the model only attends to a fixed window of the most recent tokens. This mimics the "sliding window" concept in algorithms or kernels in CNNs. It reduces memory complexity from $O(T)$ (cumulative) to $O(W)$, where $W$ is the window size. This works because semantic relevance is often local, though it may sacrifice some long-range dependency capabilities.

### vLLM and Paged Attention

One of the most significant advancements in modern LLM inference is **Paged Attention**, utilizing concepts from operating system virtual memory to solve memory fragmentation.

#### The Memory Fragmentation Problem
In standard KV Caching, memory is often over allocated or fragmented because request lengths are unknown and dynamic.

**GPU Memory Visualization (Standard):**
```text
[A][A][A][A][A][ free ][B][B][B][ free ][C][C]...
```
*Holes labeled "free" cannot be efficiently reused.*

**Issues:**
1.  **Over allocation**: Systems reserve memory for `max_seq_len` upfront.
2.  **Fragmentation**: Small gaps effectively waste VRAM.
3.  **No Sharing**: Identical prefixes (e.g., system prompts) are duplicated for every request.

#### The Paged Attention Solution
vLLM splits the KV cache into fixed-size blocks (pages), similar to how OS manages RAM.

*   **Logic**: Token positions are continuous ($1, 2, 3...$)
*   **Physical**: Blocks are scattered in GPU memory ($Block_{17}, Block_{4}, ...$)
*   **Block Table**: Maps logical blocks to physical blocks.

**Benefits:**
1.  **Zero Fragmentation**: Blocks are fixed size; any free block can be used.
2.  **Dynamic Allocation**: Allocate specific blocks only when needed.
3.  **Prefix Sharing**: Multiple requests can point to the same physical blocks for shared prompts (e.g., "Explain transformers...").

**Memory Comparison (Example):**
*Assumptions: 32 layers, $d=4096$, fp16, max_len=2048, used=300 tokens*

In worst case static allocation scenarios
*   **Standard Cache**: ~2.1 GB per request (Reserves max_len)
*   **Paged Attention**: ~40 MB per request (Allocates only used blocks)

This efficiency allows systems like vLLM to serve **10–100x** more concurrent requests (continuous batching) compared to naive implementations.

## Conclusion

Optimizing inference is not just about raw compute; it is about smart resource management. Techniques like KV Cache, Sliding Window Attention, and Paged Attention (vLLM) shift the focus from brute-force calculation to efficient memory scheduling and data retrieval. As heavy inference engines evolve, understanding these low level mechanics becomes crucial for deploying scalable LLM applications.

You can find a the code used for demonstration here [here](https://github.com/Black-fox17/llm_scratch)


**Thanks for reading!**

If you have any questions or suggestions, you can reach out to me via email at  
[Mail](mailto:ayeleru1234@gmail.com)