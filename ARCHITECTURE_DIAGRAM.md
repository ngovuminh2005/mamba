# Sơ Đồ Kiến Trúc Mamba

## 1. Tổng Quan Kiến Trúc Mamba Block

```
┌─────────────────────────────────────────────────────────────┐
│                      INPUT (B, L, D)                         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Linear Projection (in_proj)                     │
│                  D → 2*D_inner + ...                         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ├──────────┬──────────────┐
                         ▼          ▼              ▼
                    ┌────────┐ ┌────────┐    ┌─────────┐
                    │   x    │ │   z    │    │ B, C, Δ │
                    └────┬───┘ └────┬───┘    └────┬────┘
                         │          │             │
                         ▼          │             ▼
               ┌──────────────┐    │    ┌─────────────────┐
               │ Conv1D (4)   │    │    │ Selective Params│
               └──────┬───────┘    │    └────────┬────────┘
                      │            │             │
                      ▼            │             ▼
         ┌─────────────────────────┴──────────────────────┐
         │          Selective SSM Layer                    │
         │    h'(t) = A·h(t) + B(x)·x(t)                  │
         │    y(t) = C(x)·h(t) + D·x(t)                   │
         └─────────────────────┬──────────────────────────┘
                              │
                              ▼
                         ┌─────────┐
                         │  * z    │ (Gate with activation)
                         └────┬────┘
                              │
                              ▼
                   ┌──────────────────┐
                   │ Linear (out_proj)│
                   │  D_inner → D     │
                   └────────┬─────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │ OUTPUT (B, L, D) │
                   └─────────────────┘
```

## 2. Selective Scan Operation - Core Algorithm

```
Input:
  u: (B, L, D)        - Input sequence
  Δ: (B, L, D)        - Timestep (selective)
  A: (D, N)           - State matrix
  B: (B, L, N)        - Input projection (selective)
  C: (B, L, N)        - Output projection (selective)

Algorithm:
┌──────────────────────────────────────────────────────────┐
│ for each position t in sequence:                         │
│                                                           │
│   1. Discretize continuous SSM:                          │
│      Ā = exp(Δ[t] * A)                                  │
│      B̄ = (Ā - I) * A⁻¹ * B[t]                           │
│                                                           │
│   2. Update state:                                       │
│      h[t] = Ā * h[t-1] + B̄ * u[t]                       │
│                                                           │
│   3. Compute output:                                     │
│      y[t] = C[t] * h[t] + D * u[t]                      │
│                                                           │
└──────────────────────────────────────────────────────────┘

Key Innovation: B, C, Δ depend on input → Selective!
```

## 3. Mamba-2 Multi-Head Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT (B, L, D)                           │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Linear Projection (in_proj)                     │
│         D → [z, x, B, C, dt] * nheads/ngroups               │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
    ┌────────┐     ┌────────┐     ┌────────┐
    │ Head 1 │     │ Head 2 │     │ Head H │
    └───┬────┘     └───┬────┘     └───┬────┘
        │              │              │
        │   ┌──────────┴──────────┐   │
        │   │  Reshape to chunks  │   │
        │   └──────────┬──────────┘   │
        │              ▼               │
        │     ┌────────────────┐       │
        │     │ Chunk 1        │       │
        │     │ - Conv1D       │       │
        │     │ - Chunk Scan   │       │
        │     │ - Chunk State  │       │
        │     └────────┬───────┘       │
        │              ▼               │
        │     ┌────────────────┐       │
        │     │ Chunk 2        │       │
        │     └────────┬───────┘       │
        │              ▼               │
        │     ┌────────────────┐       │
        │     │ State Passing  │       │
        │     │ between chunks │       │
        │     └────────┬───────┘       │
        │              │               │
        └──────────────┼───────────────┘
                       ▼
              ┌────────────────┐
              │ Combine Heads  │
              └────────┬───────┘
                       ▼
              ┌────────────────┐
              │ RMSNorm + Gate │
              └────────┬───────┘
                       ▼
              ┌────────────────┐
              │ Output Proj    │
              └────────┬───────┘
                       ▼
              ┌────────────────┐
              │ OUTPUT (B,L,D) │
              └────────────────┘
```

## 4. Chunk-Based Processing (Mamba-2)

```
Full Sequence: [═══════════════════════════════════════════]
                         Length L

Split into Chunks:
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│ Chunk 1  │ │ Chunk 2  │ │ Chunk 3  │ │ Chunk 4  │
│ (0:256)  │ │(256:512) │ │(512:768) │ │(768:1024)│
└────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘
     │            │            │            │
     ▼            ▼            ▼            ▼
┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐
│Process  │  │Process  │  │Process  │  │Process  │
│Chunk 1  │  │Chunk 2  │  │Chunk 3  │  │Chunk 4  │
└────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘
     │            │            │            │
     │ State h₁   │ State h₂   │ State h₃   │ State h₄
     └────────────┼────────────┼────────────┤
                  ▼            ▼            ▼
            ┌──────────────────────────────────┐
            │    State Passing (forward)        │
            │  h₂ = f(h₁, dA₁)                 │
            │  h₃ = f(h₂, dA₂)                 │
            │  h₄ = f(h₃, dA₃)                 │
            └──────────────────────────────────┘

Advantages:
✓ Parallel processing within chunks
✓ Sequential state passing between chunks
✓ Memory efficient
✓ Better hardware utilization
```

## 5. Complete Mamba Language Model

```
┌─────────────────────────────────────────────┐
│         Input Token IDs: [t₁, t₂, ..., tₙ] │
└────────────────────┬────────────────────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │  Token Embedding       │
        │  vocab_size → d_model  │
        └────────────┬───────────┘
                     │
        ┌────────────▼───────────┐
        │    Mamba Block 1       │
        │  - Selective SSM       │
        │  - Gating              │
        │  - Projections         │
        └────────────┬───────────┘
                     │
        ┌────────────▼───────────┐
        │    Mamba Block 2       │
        └────────────┬───────────┘
                     │
                    ...
                     │
        ┌────────────▼───────────┐
        │    Mamba Block N       │
        └────────────┬───────────┘
                     │
        ┌────────────▼───────────┐
        │      RMSNorm           │
        └────────────┬───────────┘
                     │
        ┌────────────▼───────────┐
        │      LM Head           │
        │  d_model → vocab_size  │
        └────────────┬───────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │ Logits [v₁, v₂, ..., vᵥ]│
        └────────────────────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │  Softmax → Sampling    │
        └────────────────────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │   Next Token Prediction │
        └────────────────────────┘

Note: No positional encodings needed!
SSM naturally handles sequence order.
```

## 6. Comparison: Transformer vs Mamba

```
TRANSFORMER ATTENTION:
┌─────────────────────────────────────────────────┐
│  Q = XW_Q  K = XW_K  V = XW_V                  │
│                                                  │
│  Attention = softmax(QK^T/√d_k) × V            │
│                                                  │
│  Complexity: O(L²·d)                           │
│  Memory: O(L²)                                  │
│                                                  │
│  [t₁]──────────────┐                           │
│   │ ╲               │                           │
│  [t₂] ╲ attention   │                           │
│   │ ╲  ╲ matrix     │                           │
│  [t₃] ╲  ╲ (L×L)    │                           │
│   │ ╲  ╲  ╲         │                           │
│  [t₄] ╲  ╲  ╲       │                           │
│        All-to-all connections                   │
└─────────────────────────────────────────────────┘

MAMBA SELECTIVE SSM:
┌─────────────────────────────────────────────────┐
│  For each token t:                              │
│    h_t = f(h_{t-1}, x_t, B(x_t), C(x_t))       │
│                                                  │
│  Complexity: O(L·d²)                           │
│  Memory: O(L·d)                                 │
│                                                  │
│  [t₁]→ h₁ ─→ y₁                                │
│            ↘                                    │
│  [t₂]─────→ h₂ ─→ y₂                           │
│                  ↘                              │
│  [t₃]───────────→ h₃ ─→ y₃                     │
│                        ↘                        │
│  [t₄]─────────────────→ h₄ ─→ y₄              │
│        Recurrent connections                    │
└─────────────────────────────────────────────────┘

When L >> d: Mamba is more efficient!
```

## 7. Training and Inference Flow

```
TRAINING:
┌────────────────────────────────────────────────┐
│ Dataset (e.g., The Pile, SlimPajama)          │
└────────────────┬───────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────┐
│ Tokenization (GPT-NeoX tokenizer)              │
└────────────────┬───────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────┐
│ DataLoader (batch sequences)                   │
└────────────────┬───────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────┐
│ Forward Pass:                                  │
│  - Token Embedding                             │
│  - Stack of Mamba Blocks                       │
│  - LM Head                                     │
└────────────────┬───────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────┐
│ Compute Loss (CrossEntropy)                    │
└────────────────┬───────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────┐
│ Backward Pass:                                 │
│  - Custom CUDA kernels for gradients          │
│  - Triton kernels for efficient backprop      │
└────────────────┬───────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────┐
│ Optimizer Step (AdamW)                         │
└────────────────┬───────────────────────────────┘
                 │
                 └─── Repeat ───┘

INFERENCE (Autoregressive):
┌────────────────────────────────────────────────┐
│ Prompt: "The capital of France is"            │
└────────────────┬───────────────────────────────┘
                 │
                 ▼
        ┌────────────────┐
        │ Initial Pass   │───→ State h₀
        └────────┬───────┘
                 │
        ┌────────▼───────┐
        │ Generate t₁    │───→ Update state h₁
        └────────┬───────┘      O(1) complexity!
                 │
        ┌────────▼───────┐
        │ Generate t₂    │───→ Update state h₂
        └────────┬───────┘      Only process new token
                 │
                ...
                 │
        ┌────────▼───────┐
        │ Generate tₙ    │───→ Final state hₙ
        └────────────────┘

Key Advantage: O(1) per token generation
vs O(n) for Transformers (recompute attention)
```

## 8. Memory Layout and Optimization

```
MEMORY HIERARCHY:
┌─────────────────────────────────────────────┐
│           Global Memory (DRAM)              │
│  - Model Parameters                         │
│  - Activations                              │
│  - Gradients                                │
└────────────┬────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────┐
│         L2 Cache (shared)                   │
└────────────┬────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────┐
│         L1 Cache (per SM)                   │
└────────────┬────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────┐
│      Shared Memory (fast)                   │
│  - Tile of input/output                     │
│  - Intermediate results                     │
│  - Small matrices                           │
└────────────┬────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────┐
│       Registers (fastest)                   │
│  - Loop counters                            │
│  - Accumulated results                      │
└─────────────────────────────────────────────┘

FUSED KERNEL OPTIMIZATION:
┌──────────────────────────────────────┐
│  Without Fusion (slow):              │
│                                       │
│  GPU → Conv1D → GPU                  │
│  GPU → SSM → GPU                     │
│  GPU → Activation → GPU              │
│  GPU → Gate → GPU                    │
│  GPU → Output Proj → GPU             │
│                                       │
│  Many memory transfers! ❌           │
└──────────────────────────────────────┘

┌──────────────────────────────────────┐
│  With Fusion (fast):                 │
│                                       │
│  GPU → [Conv1D + SSM +               │
│         Activation + Gate +          │
│         Output Proj] → GPU           │
│                                       │
│  Single memory transfer! ✓           │
└──────────────────────────────────────┘
```

## 9. Distributed Training Strategy

```
DATA PARALLEL:
GPU 0: [Batch 0-7]   ──┐
GPU 1: [Batch 8-15]  ──┤
GPU 2: [Batch 16-23] ──┼──→ All-Reduce Gradients
GPU 3: [Batch 24-31] ──┘

TENSOR PARALLEL:
         Model Layer
┌──────────┬──────────┐
│  GPU 0   │  GPU 1   │  Split along feature dim
├──────────┼──────────┤
│  Col     │  Col     │  in_proj split
│  Parallel│  Parallel│
└──────────┴──────────┘
      │         │
      └────┬────┘
           │ All-Reduce
           ▼
┌──────────┬──────────┐
│  GPU 0   │  GPU 1   │
├──────────┼──────────┤
│  Row     │  Row     │  out_proj split
│  Parallel│  Parallel│
└──────────┴──────────┘

SEQUENCE PARALLEL:
Sequence: [═══════════════════════════════]
          ↓ Split
GPU 0:    [═══════]
GPU 1:            [═══════]
GPU 2:                    [═══════]
GPU 3:                            [═══════]
```

## 10. State Space Model Visualization

```
CONTINUOUS-TIME SSM:
        ┌──────┐
  x(t)  │      │  h'(t)
────────►  B   ├────────┐
        │      │        │
        └──────┘        │
                        ▼
                    ┌───────┐
              ┌─────┤   ∫   │◄──────┐
              │     └───────┘       │
              │        h(t)         │
              │                     │
              │     ┌──────┐        │
              └────►│  A   │────────┘
                    └──────┘
                        │
                    ┌───┴───┐
              ┌─────┤   C   │
              │     └───────┘
  x(t)        │
─────────┐    │
         │    │
      ┌──▼────▼──┐
      │  D (skip)│
      └──────────┘
         │
         ▼
       y(t)

DISCRETE-TIME (after discretization):
        ┌──────┐
  x[t]  │      │
────────►  B̄   ├────────┐
        │      │        │
        └──────┘        │
                        ▼
                    ┌───────┐
              ┌─────┤  +    │
              │     └───┬───┘
              │         │
              │     ┌───▼───┐
              │     │ h[t]  │
              │     └───┬───┘
              │         │
              │     ┌───▼───┐
              └─────┤  Ā    │
                    └───────┘
                        │
                    ┌───┴───┐
              ┌─────┤   C   │
              │     └───────┘
  x[t]        │
─────────┐    │
         │    │
      ┌──▼────▼──┐
      │     D    │
      └──────────┘
         │
         ▼
       y[t]
```

## Ghi Chú

- **B, L, D**: Batch size, Sequence length, Model dimension
- **N**: State dimension (d_state)
- **H**: Number of heads (nheads)
- **Chunks**: Typically size 256
- **Fusion**: Combining multiple operations into single kernel
- **State Passing**: Transferring hidden states between chunks

Các sơ đồ này giúp hiểu rõ luồng dữ liệu và cách Mamba xử lý thông tin một cách hiệu quả so với Transformers truyền thống.
