# Continuous Sign Language Translation: From MediaPipe Kinematics to Edge Transformers

## Introduction: The Kinematic Reality of Fingerspelling

Over 70 million deaf individuals worldwide use sign language as their primary mode of communication. In American Sign Language (ASL), fingerspelling plays an indispensable role: it is how signers communicate proper nouns, technical terms, addresses, names, and words that lack a standardized semantic sign.

Yet, building computer vision models to translate continuous fingerspelling from raw camera video into written text is notoriously difficult. Unlike isolated gesture classification (where a model classifies a single static hand pose from a cropped bounding box), continuous fingerspelling involves:
1. **Subtle and rapid joint articulators** occurring at 3 to 6 characters per second.
2. **High temporal occlusion**, where moving fingers cross over one another relative to the camera plane.
3. **Severe signer variance**, including left-handed vs. right-handed signers, differences in signing speed, and shifting positions within the camera frame.
4. **Edge latency constraints**: An assistive communication system is only useful if it runs in real time directly on a mobile or edge device without sending massive video streams over high-latency networks.

When I started engineering an end-to-end fingerspelling translation system, I wanted to avoid brute-forcing the problem with gigantic 3D convolutional video networks (like I3D or SlowFast) that consume gigabytes of memory and fail to run in real time. Instead, I designed a pipeline grounded in first principles: **extract raw geometric landmarks, enforce geometric symmetries mathematically, compress temporal frames via strided 1D convolutions, and model character sequences with an autoregressive Transformer decoder**.

In this article, I will break down the mathematical formulations, architecture, and engineering optimizations that make continuous ASL translation feasible on edge hardware.

---

## The Raw Data & Feature Slicing

A standard video is composed of millions of raw pixels per second: $T \times H \times W \times 3$. Passing raw video pixels into a sequence model forces the neural network to spend the majority of its parameter capacity learning invariant representations of lighting, background clutter, clothing textures, and skin tones, factors completely unrelated to the linguistic content of the gesture.

Using **MediaPipe Holistic**, we isolate the kinematic topology of the signer into 543 spatial coordinates in 3D:
- Face mesh: $468$ landmarks
- Body pose: $33$ landmarks
- Left hand: $21$ landmarks
- Right hand: $21$ landmarks

Each landmark contains $(x, y, z)$ coordinates normalized relative to the camera viewport.

```
Total Landmarks = 468 + 33 + 21 + 21 = 543 3D Keypoints
```

### Eliminating Redundancy

Fingerspelling does not require face mesh deformations or lower-body pose tracking. To minimize input dimensionality while preserving necessary context, we slice the landmark tensor down to **48 essential landmarks**:
- **Right Hand** ($21$ keypoints: wrist, thumb, index, middle, ring, pinky joints)
- **Left Hand** ($21$ keypoints)
- **Upper Pose** ($6$ keypoints: left/right shoulders, elbows, and wrists; indices `[11, 12, 13, 14, 15, 16]`)

Tracking the upper pose keypoints provides the spatial reference anchor: it informs the network of the signer's torso orientation and resting arm position.

With 48 landmarks and 3 spatial coordinates each, we reduce our per-frame input dimensionality from hundreds of thousands of RGB pixels down to exactly:

$$D = 48 \times 3 = 144 \text{ continuous features}$$

For a standard sequence length of $T = 128$ frames, our entire input representation is a compact matrix of shape:

$$(T, D) = (128, 144)$$

---

## Geometric Invariance: Handling Handedness & Drift

One of the largest failure modes in gesture recognition is hand orientation mismatch. A left-handed signer executing the letter "B" presents a mirror image of a right-handed signer. If a model must learn separate parameter manifolds for left- and right-handed signers, its effective sample complexity doubles.

To fix this, we apply **deterministic orientation normalization** before the model ever sees the data.

### 1. Dominant Hand Detection via Missing Value Ratios

In real video streams, signers rarely fingerspell with both hands simultaneously; the non-dominant hand typically rests out of frame or hangs loosely by their side. When a hand is out of frame, MediaPipe outputs `NaN` values.

We quantify the number of missing frames for each hand:

$$\text{NaN}_{\text{right}} = \sum_{t=1}^T \mathbb{I}(\text{RightHand}_t = \text{NaN})$$

$$\text{NaN}_{\text{left}} = \sum_{t=1}^T \mathbb{I}(\text{LeftHand}_t = \text{NaN})$$

The hand exhibiting fewer missing frames is designated as the active/dominant hand:

$$\text{Active Hand} = \begin{cases} \text{Left}, & \text{if } \text{NaN}_{\text{left}} < \text{NaN}_{\text{right}} \\ \text{Right}, & \text{otherwise} \end{cases}$$

### 2. Reflection Invariance via Horizontal Flip

If the signer is left-handed, we reflect their horizontal spatial coordinates across the vertical axis:

$$x_{\text{normalized}} = 1.0 - x$$

This mirrors the left hand into the canonical coordinate frame of a right hand. Suddenly, the neural network only needs to learn a single canonical representation of every letter.

```python
# From src/data/preprocessing.py
if rnans > lnans:
    # Left hand is dominant: mirror coordinates across x-axis
    hand = lhand
    pose = lpose
    hand_x = hand[:, 0 * num_hand_coords : 1 * num_hand_coords]
    hand = tf.concat([1.0 - hand_x, hand[:, y_slice], hand[:, z_slice]], axis=1)
```

### 3. Spatial Centering and Unit-Variance Standardization

Signers move closer to or further from the camera, and drift across the screen. To make our features scale- and position-invariant, we subtract the instantaneous spatial centroid $\mu_t$ and divide by the standard deviation $\sigma_t$ for each frame:

$$\hat{\mathbf{p}}_t = \frac{\mathbf{p}_t - \mu_t}{\sigma_t + \epsilon}$$

where $\mu_t = \frac{1}{K}\sum_{k=1}^K \mathbf{p}_{t,k}$.

This guarantees that whether a hand is in the top-left corner or center-right of the video, its coordinate representation is zero-centered and bounded.

---

## Temporal Downsampling via Strided 1D Convolutions

Now we have a clean sequence of shape $(B, 128, 144)$. 

Standard full self-attention scales quadratically with sequence length: $\mathcal{O}(T^2)$. A sequence of 128 frames means $128 \times 128 = 16,384$ attention score computations per head per layer. Furthermore, adjacent video frames at 30 FPS exhibit massive temporal redundancy: the hand barely moves between frame $t$ and frame $t+1$.

To compress the temporal axis while extracting local kinematic trajectory vectors (velocity and curvature of the joints), we pass the raw features through **three cascaded 1D convolutional layers with stride 2**:

```python
# From src/models/embedding.py
class LandmarkEmbedding(layers.Layer):
    def __init__(self, num_hid=64):
        super().__init__()
        # Layer 1: 128 frames -> 64 steps
        self.conv1 = layers.Conv1D(num_hid, kernel_size=11, strides=2, padding="same", activation="relu")
        # Layer 2: 64 steps -> 32 steps
        self.conv2 = layers.Conv1D(num_hid, kernel_size=11, strides=2, padding="same", activation="relu")
        # Layer 3: 32 steps -> 16 steps
        self.conv3 = layers.Conv1D(num_hid, kernel_size=11, strides=2, padding="same", activation="relu")
```

### Why a Kernel Size of 11?

A kernel size of 11 spans approximately one-third of a second of video at 30 FPS. This is roughly the exact temporal duration of a single fingerspelled phoneme/letter transition. 

By applying three successive stride-2 convolutions:

$$T = 128 \xrightarrow{\text{stride 2}} 64 \xrightarrow{\text{stride 2}} 32 \xrightarrow{\text{stride 2}} 16$$

We compress $128$ raw time steps down to $16$ dense spatio-temporal tokens of hidden dimension $d_{\text{model}} = 64$. 

The attention matrix computation inside the Transformer shrinks from:

$$128^2 = 16,384 \quad \longrightarrow \quad 16^2 = 256 \text{ operations}$$

That is a **$64\times$ reduction in attention computational complexity**, making the model lightning-fast for edge devices.

---

## The Sequence-to-Sequence Architecture

The core of the system is an encoder-decoder Transformer:

```
[ Landmark Input (B, 128, 144) ]
               │
               ▼
   [ 3x Conv1D Strided Emb ]  ───► Output shape: (B, 16, 64)
               │
               ▼
   [ 4x Transformer Encoder ] ───► Multi-Head Self-Attention + FFN
               │
               │ Memory Context (B, 16, 64)
               ▼
   [ Causal Cross-Decoder ]   ◄─── Target Tokens: (B, L) shifted right
               │
               ▼
   [ Character Classification ] (62 Classes: 59 chars + pad + start + end)
```

### 1. Transformer Encoder

The encoder consists of 4 stacked layers. Each layer contains:
- Multi-Head Self-Attention with 2 attention heads and $d_k = 64$.
- Pre-Layer Normalization and residual skip connections.
- Feed-Forward sublayers expanding to dimension $d_{\text{ff}} = 128$ with ReLU activations.

Because the encoder tokens represent compressed temporal video chunks, self-attention allows the model to capture co-articulation effects, where the formation of an upcoming letter influences the hand shape of the current letter.

### 2. Transformer Decoder with Causal Masking

The decoder receives the shifted-right character tokens (starting with `<` token) and generates text autoregressively.

1. **Masked Self-Attention**: Prevents character position $i$ from attending to future tokens $j > i$.
2. **Cross-Attention**: Queries ($Q$) come from the target text tokens, while Keys ($K$) and Values ($V$) come from the encoder's video representations:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

This allows the model to align each generated character with the corresponding temporal segments in the video trajectory.

---

## Loss Function & Evaluation Metric

### 1. Label-Smoothed Masked Cross-Entropy

During training, shorter phrases are padded with special token `P` (`pad_token_idx = 59`). Computing loss over padding tokens degrades gradients. We explicitly mask out padding indices and apply label smoothing ($\alpha = 0.1$):

```python
loss = tf.keras.losses.categorical_crossentropy(
    target_one_hot, 
    predictions, 
    label_smoothing=0.1
)
mask = tf.cast(tf.not_equal(target_indices, PAD_TOKEN_IDX), dtype=loss.dtype)
loss = tf.reduce_sum(loss * mask) / tf.reduce_sum(mask)
```

### 2. Normalized Total Levenshtein Distance

Word Error Rate (WER) is an inadequate metric for fingerspelling because missing a single character in a name (e.g., "Micheal" vs "Michael") should not penalize the entire phrase as a total failure.

Instead, we evaluate models using the **Normalized Total Levenshtein Distance**:

$$\text{Metric} = \frac{\sum_{i=1}^N \mathcal{D}(P_i, T_i)}{\sum_{i=1}^N |T_i|}$$

where $\mathcal{D}(P_i, T_i)$ is the edit distance (minimum insertions, deletions, and substitutions) between predicted string $P_i$ and ground truth $T_i$, and $|T_i|$ is the target phrase length.

---

## Edge Optimization: Exporting to TensorFlow Lite

A trained deep learning model in Python is tethered to heavy framework dependencies (`torch`, `tensorflow`, `cuda`). For real-world assistive tech, the model must run inside a native mobile app (iOS or Android) without network latency.

We encapsulate the entire pipeline (feature slicing, orientation flip, spatial normalization, temporal convolution, and autoregressive generation) into a single standalone **TensorFlow Lite (`model.tflite`)** graph:

```python
# From src/inference/tflite_export.py
class TFLiteModel(tf.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 144], dtype=tf.float32, name="inputs")])
    def __call__(self, inputs):
        # 1. On-device preprocessing
        x = pre_process(inputs)
        x = tf.expand_dims(x, axis=0)
        
        # 2. Greedy autoregressive decode loop
        outputs = self.model.generate(x, max_len=100)
        return {"outputs": outputs}
```

The resulting model:
- Weighs just **~39 MB**.
- Executes in **under 30ms** per gesture sequence on modern mobile NPUs.
- Exposes a unified serving signature requiring zero external preprocessing libraries.

---

## Summary & Key Takeaways

1. **Don't feed raw pixels to sequence models**: Extracting kinematic landmarks with MediaPipe Holistic cuts feature dimensionality by orders of magnitude while stripping away irrelevant background noise.
2. **Inject physical priors into data pipelines**: Automating dominant hand detection and applying $1 - x$ horizontal reflection eliminates handedness variance, halving the model's required hypothesis space.
3. **Temporal striding tames Transformer complexity**: Using three $1\text{D}$ convolutions with stride 2 compresses $128$ frames to $16$ latent steps, slashing self-attention compute by $64\times$.
4. **Design for the edge from day one**: Packaging preprocessing directly into the computational graph enables clean `.tflite` deployment with zero-dependency native execution.

You can explore the full code, training pipelines, and exported models on GitHub: [github.com/Black-fox17/ASL](https://github.com/Black-fox17/ASL).
