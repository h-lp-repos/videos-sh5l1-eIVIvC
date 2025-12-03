# Self-Attention & Latency Visualizations

The purpose of these demos is to provide visual support for the theoretical explanations of:

- Self-attention in Transformer models
- The relationship between attention and latency
- Quadratic scaling with respect to sequence length

## Setup

### 1. Open a terminal in the root of this repository

---

### 2. Create and activate a virtual environment

**Using `venv` (recommended):**

```bash
python3 -m venv .venv
source .venv/bin/activate        # Linux / macOS
.\.venv\Scripts\activate        # Windows PowerShell
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Demos

### Video 1: Self-Attention Visualization

```bash
python video-1-self-attention/self_attention_visualization.py
```

---

### Video 2: Attention vs Latency Benchmark

```bash
python video-2-latency/attention_latency_benchmark.py
```

## Troubleshooting

* **Slow first execution**

  The first run will download model weights from Hugging Face.
  Subsequent runs will be much faster.