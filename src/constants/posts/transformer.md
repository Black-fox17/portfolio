# Why Attention Beats Recurrence: A Deep Dive from LSTM to Transformer

## Introduction: Sequential Modeling in NLP

The field of NLP has experienced a series of evolutions and major breakthroughs. The most visible of these, and the hottest topic right now, is Large Language Models (LLMs). These models can perform a series of many tasks, ranging from text translation to question answering, sentiment analysis, you name it. We can call it a knowledge base that understands languages so well, but this wasn't just something that came to be in 2017. In fact, this is a result of years of research dating back to the 1950s but didn't really take off due to limitations of computing power (Note this). Major breakthroughs came in the 90s when we could compute statistical algorithms and they yielded results.

You see, Artificial Intelligence/Machine Learning, Mathematics and Statistics go hand in hand. Most statistical algorithms used for modeling were statistics that had been discovered 200 years ago, but with the advancement in compute power, we were able to perform heavy computation using these algorithms for prediction.

Deep Neural Networks were part of these statistical models, and it really took off in the community of researchers then, especially with the introduction of backpropagation popularized by David E. Rumelhart et al. But you see, the idea of backpropagation was based on the chain rule derived by Gottfried Wilhelm Leibniz in 1673. Artificial Neural Networks follow the same format: given an input $x$, a model to be $f(x)$, we try to predict $y$, i.e., $y = f(x)$.

There were many innovations and architectures derived from this, and all of it depends on how you model $f(x)$. It is the backbone behind the whole industry of Deep Learning right now. RNN (Recurrent Neural Network) was part of the innovation that stems from it. Unlike Feedforward Neural Networks which operate by $y = f(x)$ on a single input, RNN borrows from the idea of recurrence and changes the equation format to:

$$ y_t = f(x_t, h_{t-1}) $$

where $x_t$ is the input from a previous step. This simple idea here is made possible by a shortcut connection, a vital part of our RNN architecture where input from a previous time step is fed back as an input to the next time step. This enables RNNs to capture temporal dependencies and patterns within sequences.

It was successfully applied in sequence tasks like connected handwriting recognition, speech recognition, natural language processing, and neural machine translation. This was the foundation for our sequential modeling history.

Traditional RNNs suffer from the vanishing gradient problem, which limits their ability to learn long-range dependencies. This was fixed by variant architectures like LSTM (Long Short Term Memory) and GRU (Gated Recurrent Units). LSTM, introduced in the 90s, became the go-to architecture for many transduction models then, being the backbone of seq2seq engines like Google Translate. Meanwhile, the 2015 ImageNet ISLVRC competition was won by ResNet (Residual Neural Network), which introduced residual connections to train extremely deep networks—a concept distinct from the recurrence found in LSTMs. While this revolutionized sequential modeling, LSTM still has a major flaw: it processes text in sequence and couldn't manage long text windows, which renders it useless for understanding meaningful points in a text. With Moore's Law, computation power increased drastically over the years. With the popularization of parallel computing and enough compute power, one might really ask the question if LSTM is really the best approach to sequential modeling.

Attention was a popular concept, or rather I say notion, in Machine Learning which involves parts in a sequence relating to each other and has been around for some time. This would prove to be helpful in modeling long range sequences, and the word at the beginning could relate with the word at the end. For instance: "Michael Jordan is a great basketballer, He really is the ______". Unlike in RNNs, which encode a strong inductive bias toward recent tokens due to sequential processing, self-attention allows each token to directly reference any other token in the context window, regardless of distance. With the release of "Attention Is All You Need" paper in 2017 where they formalized the scaled dot product attention matrix and among many other layers of their transformer architecture which supports parallel computing, it really became the go-to approach in all AI/ML fields: Computer Vision, NLP, you name it.

In this blog, we will be looking at LSTM, a variant architecture of RNN, and Transformer, an architecture inspired by attention.

## LSTM Architecture

LSTM is meant to fix the long range dependency issues of RNN. One might ask how? RNN has a repeating module in its layers. LSTM handles recurrence, or let's say passing information input from a previous state, through a special gating mechanism which controls information flows.

### The Four Gates

**Input Gate ($i$):** Controls how much new information to let in.
$$ \sigma(\dots) \rightarrow \text{values between } 0 \text{ and } 1 $$
$0$ means "block everything", $1$ means "let everything through".

**Forget Gate ($f$):** Controls what to discard from memory.
$$ \sigma(\dots) \rightarrow \text{values between } 0 \text{ and } 1 $$
$0$ means "forget completely", $1$ means "remember everything".

**Cell Gate ($g$):** Creates new candidate information.
$$ \tanh(\dots) \rightarrow \text{values between } -1 \text{ and } 1 $$
The actual content to potentially add to memory.

**Output Gate ($o$):** Controls what to output from the cell.
$$ \sigma(\dots) \rightarrow \text{values between } 0 \text{ and } 1 $$
Filters what part of the cell state becomes the hidden state.

### State Updates

$$ c_{next} = f \cdot c + i \cdot g $$
Old memory ($c$) $\times$ forget gate ($f$) for what to keep.
New info ($g$) $\times$ input gate ($i$) for what to add.
Result is the updated cell state.

$$ h_{next} = o \cdot \tanh(c_{next}) $$
Apply $\tanh$ to cell state to normalize to $[-1, 1]$.
Multiply by output gate ($o$) to filter what to expose.
Result is the Hidden state (what the network "sees").

In the training sample on the GitHub repo, we build a character level language model with LSTM and it follows the principle of the gating mechanism. We could trace one training step with sequence "hello" from our training code:

```python
# From RNN/train.py
    for t in range(x.size(1)):
        logits, state = model(x[:, t], state)
        loss += F.cross_entropy(logits, y[:, t])
```

Here is a visual representation:

![LSTM Character-Level Language Model Architecture. From bottom to top: Input character indices (e.g., 5, 12, 8). Embedding Layer converting indices to dense vectors. LSTM Layer 1 with Input, Forget, Cell, and Output gates showing information flow, with hidden state h1 and cell state c1. LSTM Layer 2 with similar structure. Output Layer projecting to vocabulary size with probability distribution. Output predicting the next character. Arrows show state flow through time steps.](/blog/lstm.png)

With this gating mechanism, we could see why it was more powerful than the traditional RNN.

## Transformer Architecture and Self-Attention

Self-attention fixes relation. It proposes the fact that given a sequence, we can compute the score at which each word in the sequence relates with each other. This disregards the notion of using recent information to predict the next token, but rather a token at the beginning of our sequence contributes to the prediction of the next token.

This is the fundamental aspect of Transformer, the modern day architecture for sequential modeling.

In the "Attention Is All You Need" paper, they formalize the notion of scaled dot attention matrix which is:
$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$

![Transformer Architecture Diagram](/blog/transformer.png)

Where $Q$ is the Query, $K$ is the Key, and $V$ is the Value.

The idea is that for each token in our sequence, we compute a query vector, key vector, and value vector. We compute the attention weight of how this token relates with each other. This is done by the dot matrix product of the query and key vectors divided by the square root of the dimension size. In an autoregressive task where we generate the next token, we don't want "cheating" in our model training; i.e., we want to limit what the model can see at a time in a sequence during training. So a token in a sequence only relates to itself and tokens before it. This is done by the Masking technique where the upper triangle of our dot product matrix is filled and the lower part are replaced by $-\infty$. This ensures during softmax the sum is $1$ and the lower triangle matrix are $0$s, and then we dot product with value $V$. Masking is required to prevent information leakage during autoregressive training. Without it, the model can attend to future tokens, producing unrealistically low training loss and invalid generalization.

Here is how it looks in code:

```python
# From attention/models/GPT.py
    def forward(self, x, mask = True):
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        q, k, v = self.split_heads(q), self.split_heads(k), self.split_heads(v)
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask:
            mask = torch.triu(torch.ones(attn.size(-2), attn.size(-1), device=attn.device), diagonal=1).bool()
            attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).contiguous()
        out = out.view(out.size(0), out.size(1), self.d_model)
        out = self.W_o(out)
        return out
```

Note: Cross attention, a variation of self-attention, does not make use of masking. Masking ensures later tokens do not influence preceding tokens.

In our transformer, we make use of Multi-Head Attention which is multiple attention heads performing the same similar score operation but with the benefit of powerful computation. This could be run in parallel, making sure our model learns important details in how tokens relate to each other, basically just like a human learning about grammatical notions of a language. For the multi-head attention, it is really nothing fancy, just size manipulation.

There are certain layers in our transformer mostly to help with stabilizing training. These are Layer Normalization:
$$ \text{LayerNorm}(x)_i = \gamma_i \cdot \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta_i $$

It helps keep the network numerically sane so attention and FFNs can actually learn.

And there is also the Feed Forward layer. This is really something we have mentioned earlier. Like we said, a FFN doesn't concern itself with the environment; it focuses on transforming input to values so the computation done here doesn't influence the attention between tokens. It comes after attention and focuses on representing individual tokens in their own representation. It turns "raw attended context" into usable features.

It enables abstraction, pattern extraction, and decision-ready representations. It is where most of the model parameters live, so to say.

Well, in simple terms, this is Transformer in a nutshell.

## How it performs against LSTM

We have a code repo [here](https://github.com/Black-fox17/llm_scratch) showing the demonstration of both for a character language model. Though in the modern world we make use of a Byte Pair Tokenizer for tokenization, not character, this was just for the sake of an experiment. It is worth noting Transformer learns quickly compared to LSTM. It could be able to understand how characters relate with each other in forming words but struggle with positions. This is because self-attention is inherently permutation-invariant; without positional information, the model cannot distinguish token order. Though LSTM struggles with this a bit and learns gradually and could generate a comprehensible sentence at a certain epoch after training. This was partly due to dataset size, mode of tokenization, and training epochs.

It is worth noting when we mention how attention is computed and the whole architecture of Transformer, we didn't mention some things about order despite the fact it is run parallel. So how does the model learn to differentiate between "The cat is sitting on the chair" from "The chair is sitting on the cat"? This is done with Positional Embedding. You can check my previous blog on this [Link to my previous blog].

And with the benefit of parallel computation, we can generate better results in the long run and way better long-range dependency compared to LSTM.

LSTM processes sequences sequentially, making it $O(N)$ but difficult to parallelize, and its performance degrades on long sequences (weak beyond ~100s tokens). The Transformer, while $O(N^2)$ due to the attention matrix, is fully parallelizable and scales predictably with hardware. This architectural shift allows Transformers to handle global dependencies across the entire context window efficiently, which is the key reason for their dominance.

Also, it is worth noting the fact of how important masking is for this autoregressive generation. Masking is required to prevent information leakage during autoregressive training. Without it, the model can attend to future tokens, producing unrealistically low training loss and invalid generalization.

## Sampling

When it comes to generating text, you have to know certain things which are basically temperature, top k, and top p. These are parameters passed to prevent a greedy approach to model inference. Our input is passed through a linear layer which outputs a vector of probability logits. This corresponds to the likelihood of the token to be chosen. Without temperature, top_k, top_p, our model uses argmax which just returns the highest prob of the token. So for each generation, the model produces similar output.

To introduce a set of diversity to our model, the concept of **temperature** was introduced. This reduces power for the highest prob token and tones the logits down. Then we use `torch.multinomial` to do random selection. This in simple terms, according to PyTorch website, is for a tensor `[0.3, 0.6, 0.1, 0.1, 0.1]` corresponding to `[a, b, c, d, e]`, `torch.multinomial` ensures `b` has a 60% chance of being chosen and `a` has a 30% chance of being chosen. This introduces diversity and creativity in our model output.

**Top k** basically follows the concept of giving a value of $k$ (let's say 10). It picks the top 10 probability tokens then renormalized their probability, and choose from this.

**Top p** (nucleus sampling) given $p$ chooses a small set of tokens probs that their cumulative sum equal to $p$ then sample from this.

While this does introduce creativity to our model generation, it poses a greater risk which is higher values of these could lead the model picking tokens which doesn't correlate to the context of the sequence. At least lower probs are lower for a reason.

So you have to be careful with your approach and decision in choosing this.

## In Modern Context

Although the idea about sequential modeling, especially transduction models which are the encoder-decoder models, has been to achieve longer context range dependency (i.e., longer text window to process), Transformer beat LSTM. This is what led to how we were able to create Question Answering bots as long as we could fit the whole texts for the model to process in the same context window. With the help of some Supervised Fine Tuning (SFT), the model could produce a meaningful response to the user.

Modern techniques have spiraled out to address the bottleneck of the transformer by fixing its positional embedding as introduced by the authors. Several researchers have come up with a better way to do this, and it will increase context window over inference. Not only this, certain parts of our architecture have been changed to improve stability and training performance. In our traditional transformer, we made use of GeLU activation layer; modern transformers commonly use GeLU or SiLU activations rather than ReLU. Many modern architectures replace LayerNorm with RMSNorm due to its simplicity and improved numerical stability, though LayerNorm is still widely used. More parameters so the model could learn more and more patterns. Using KV Cache for faster inference.

## Conclusion

The evolution from LSTM to Transformer marks a pivotal shift in how we approach sequential modeling, from sequential recurrence to parallel attention. While LSTMs laid the groundwork for handling state and memory, Transformers, through the mechanism of self-attention, unlocked the ability to model global dependencies at scale. This architecture has not only revolutionized NLP but has also been adapted for computer vision (e.g., Vision Transformers or ViTs), audio processing, and more, proving that attention is indeed a powerful inductive bias.

This raises an open question: is scaling attention-based models with better positional encodings sufficient, or do we need fundamentally new inductive biases beyond attention to achieve Artificial General Intelligence?

**Thanks for reading!**

If you have any questions or suggestions, you can reach out to me via email at  
[Mail](mailto:ayeleru1234@gmail.com)