# HDL_Verification_CA_Extra
Fine-tuning Llama 3.2 1B using LoRA and 4-bit quantization to automate SystemVerilog Assertion (SVA) generation from Verilog code. Designed for formal verification workflows (ABV) as part of the Functional HDL Verification course.

# Automated Hardware Assertion Generation using Fine-tuned Llama 3.2

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

This repository contains the implementation for the final assignment of the **Functional HDL Verification** course. The project focuses on fine-tuning the **Llama 3.2 1B** Large Language Model (LLM) to automatically generate **SystemVerilog Assertions (SVA)** from HDL (Verilog) code. These assertions are crucial for formal verification methods such as Model Checking and Assertion-Based Verification (ABV).

---

## 📋 Table of Contents
- [Introduction](#introduction)
- [Course Information](#course-information)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [Model Architecture](#model-architecture)
  - [Fine-Tuning Techniques](#fine-tuning-techniques)
- [Installation & Usage](#installation--usage)
- [Results](#results)
- [Acknowledgments](#acknowledgments)

---

## 📖 Introduction

In the complex landscape of hardware design, ensuring functional correctness is paramount. **Assertion-Based Verification (ABV)** is a powerful technique where design properties are formally described as assertions. However, manually writing these assertions is time-consuming and error-prone.

This project leverages the power of Generative AI to automate this process. By fine-tuning a lightweight yet powerful LLM (Llama 3.2 1B), we aim to translate Verilog design descriptions directly into syntactically correct SystemVerilog Assertions, thereby accelerating the verification cycle.

---

## 🎓 Course Information

*   **Course:** Functional HDL Verification
*   **Instructor:** Dr. Siamak Mohammadi
*   **Institution:** University of Tehran
*   **Semester:** Fall 2025 (Pauiz 1404)
*   **Teaching Assistants:** Behzad Jannati, Mahsa Abarghani Aghdam

---

## 📊 Dataset

We utilized the **VERT (Verilog to SystemVerilog Assertions)** dataset for training and evaluation.
*   **Source:** [AnandMenon12/VERT](https://github.com/AnandMenon12/VERT)
*   **Structure:** The dataset consists of pairs containing:
    *   `Code`: The Verilog HDL implementation.
    *   `Assertion`: The corresponding SystemVerilog Assertion (SVA).
    *   Metadata: Information about clock signals and synchronous/asynchronous behavior.
*   **Preprocessing:** Data was formatted into an "Instruction Following" format to effectively train the model as a hardware verification expert.

---

## ⚙️ Methodology

### Model Architecture
We selected **Meta's Llama 3.2 1B** model. This model offers an excellent balance between performance and computational efficiency, making it suitable for fine-tuning on consumer-grade GPUs (like the Tesla T4 provided in Google Colab).

### Fine-Tuning Techniques
To achieve efficient training, we employed **Parameter-Efficient Fine-Tuning (PEFT)** strategies:

1.  **Quantization:**
    *   Used `bitsandbytes` for **4-bit Normal Float (NF4)** quantization.
    *   Reduced model memory footprint while maintaining inference quality.

2.  **LoRA (Low-Rank Adaptation):**
    *   Instead of full fine-tuning, we froze the base model weights and injected trainable rank decomposition matrices.
    *   **Rank (r):** 16
    *   **Alpha:** 32
    *   **Target Modules:** `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.

3.  **Training Configuration:**
    *   **Framework:** Hugging Face `TRL` (SFTTrainer).
    *   **Optimizer:** Paged AdamW (32-bit).
    *   **Precision:** Mixed precision (FP16).
    *   **Gradient Accumulation:** Used to simulate larger batch sizes.

---

## 🚀 Installation & Usage

To replicate this project, you can use Google Colab or a local environment with GPU support.

### Prerequisites
*   Python 3.10+
*   Hugging Face Account (with access token)
*   Access approved for `meta-llama/Llama-3.2-1B`

### Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME

# Install dependencies
pip install -r requirements.txt
# Key libraries: transformers, datasets, peft, bitsandbytes, trl, accelerate
