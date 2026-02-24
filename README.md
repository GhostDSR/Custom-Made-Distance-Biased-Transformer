# Distance-Biased Transformer (DB-Transformer)

A novel, lightweight Transformer architecture that replaces complex sinusoidal positional encodings with an intuitive **Distance-Biased Attention Mechanism**, applied here to the MNIST dataset.

## Project Overview
[cite_start]The standard Transformer architecture relies solely on attention mechanisms, dispensing with recurrence and convolutions entirely[cite: 17]. [cite_start]However, because self-attention processes all tokens simultaneously, it requires positional encodings to understand sequence order[cite: 165]. [cite_start]The original implementation uses mathematically complex sine and cosine functions[cite: 169]. 

This project introduces the **DB-Transformer**. Instead of injecting positional data into the input embeddings, this model modifies the self-attention calculation itself. It applies a learnable **distance penalty** to the attention scores, forcing the model to inherently prioritize local context (neighboring tokens) while retaining the ability to learn long-range dependencies.

### Key Features
* **No Positional Encodings:** Completely removes the standard sinusoidal/learned embedding steps.
* **Simplified Math:** Uses a raw physical distance matrix scaled by a learnable parameter ($\lambda$).
* **1D Sequence Vision:** Processes images by flattening them into ultra-long 1D sequences (e.g., 784 tokens for MNIST) without 2D convolutions.

## The Architecture (Distance Bias)
In standard Scaled Dot-Product Attention, scores are calculated as:
$Attention(Q,K,V) = softmax\left(\frac{QK^{T}}{\sqrt{d_{k}}}\right)V$

In this proposed **Distance-Biased Attention**, we compute a Distance Matrix $D$ where $D_{i,j} = |i - j|$ and penalize distant tokens:
$Attention_{DB}(Q,K,V) = softmax\left(\frac{QK^{T}}{\sqrt{d_{k}}} - \lambda D\right)V$

## Results (Proof of Concept)
The model was evaluated on the **MNIST** dataset. The $28 \times 28$ images were flattened into sequences of 784 tokens.

| Metric | Score | Note |
| :--- | :--- | :--- |
| **Accuracy** | ~79% | Reached in just 10 epochs |
| **Sequence Length** | 784 | Processed natively as 1D text |
| **Parameters** | ~250k | Highly lightweight |

**Analysis:** While standard CNNs reach 99% on MNIST by utilizing hardcoded 2D spatial biases, achieving 79% in just 10 epochs on a 1D sequence of length 784 is a highly successful proof-of-concept. It demonstrates that the distance-penalty successfully replaces sinusoidal positional encodings and guides the network to prioritize local context.
