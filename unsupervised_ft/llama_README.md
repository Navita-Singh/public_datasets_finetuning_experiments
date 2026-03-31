# 📘 Fine-Tuning LLaMA-3 (8B) with Unsloth + LoRA

This project demonstrates how to fine-tune a large language model (Meta
LLaMA) using parameter-efficient fine-tuning with LoRA, implemented
through Unsloth and Hugging Face tooling.

The goal of this setup is to train a lightweight adapter on an
instruction dataset efficiently, using minimal GPU memory while
maintaining stable training behavior.

------------------------------------------------------------------------

## 🚀 Overview

This project fine-tunes LLaMA-3 8B (4-bit quantized) on an instruction
dataset using:

-   LoRA (Low-Rank Adaptation)
-   4-bit quantization (QLoRA-style setup)
-   Fully controlled tokenization pipeline
-   Stable PyTorch training loop (no dynamic packing issues)

The implementation avoids unstable automatic dataset processing and
instead explicitly handles tokenization and label creation to ensure
deterministic training behavior.

------------------------------------------------------------------------

## 🧠 Why This Approach?

Fine-tuning large language models is often unstable when using automatic
trainers due to:

-   inconsistent sequence lengths
-   implicit tokenization logic
-   dynamic padding / packing conflicts
-   loss mismatch between inputs and labels

This project solves these issues by:

- Fully manual tokenization\
- Explicit label construction\
- Fixed sequence length padding\
- Removal of ambiguous trainer preprocessing

This ensures that input IDs and label tensors always match exactly,
preventing cross-entropy shape mismatches during training.

------------------------------------------------------------------------

## ⚙️ Key Components

### Model Loading

We load a quantized LLaMA-3 8B model in 4-bit precision to reduce GPU
memory usage.

------------------------------------------------------------------------

### LoRA Adaptation

We apply LoRA:

-   Rank (r): 16\
-   Target modules: attention + MLP projections\
-   Dropout: 0

LoRA enables efficient fine-tuning by training only small adapter
matrices instead of full model weights.

------------------------------------------------------------------------

### Dataset

We use the argilla/farming dataset, containing:

-   instruction
-   response

Each example is converted into:

### Instruction:

... \### Response: ...

------------------------------------------------------------------------

### Tokenization Pipeline

Each sample is manually tokenized:

-   truncation to max length (512)
-   padding to fixed length
-   labels = input_ids

This ensures deterministic tensor shapes.

------------------------------------------------------------------------

### Training Setup

-   Batch size: 1\
-   Gradient accumulation: 8\
-   Learning rate: 2e-4\
-   Max steps: 60\
-   Optimizer: AdamW 8-bit

------------------------------------------------------------------------

## 🧪 Why Not Use Packing?

Packing improves speed but introduces:

-   label misalignment
-   token boundary issues
-   shape mismatch errors

It is disabled for stability.

------------------------------------------------------------------------

## 📊 Expected Behavior

-   steady loss decrease\
-   no shape mismatch errors\
-   stable GPU utilization

------------------------------------------------------------------------

## 🔧 Troubleshooting

If errors occur:

-   disable packing\
-   ensure labels = input_ids\
-   avoid mixing auto and manual preprocessing

------------------------------------------------------------------------

## 🚀 Future Improvements

-   ChatML formatting\
-   response-only loss masking\
-   evaluation loop\
-   checkpoint saving\
-   export to vLLM

------------------------------------------------------------------------

## 📌 Summary

This project provides a stable fine-tuning pipeline for LLaMA-3 using
LoRA and Unsloth, prioritizing correctness and reproducibility over
aggressive optimization.
