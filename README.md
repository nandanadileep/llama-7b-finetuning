# LLaMA 7B Fine-Tuning

This project is done to intentionally change the behavior of a general-purpose LLaMA-7B model so it reliably follows a specific instruction style and response structure. Fine-tuning is used because prompting alone cannot enforce consistent behavior, output format, or domain patterns at scale. By training adapter layers, the model internalizes task-specific reasoning instead of repeatedly relying on long prompts. This reduces inference cost, latency, and randomness while improving controllability. The result is a deployable, ownable model variant suitable for real applications rather than experimentation.

## Demo

<p align="center">
  <a href="https://github.com/nandanadileep/llama-7b-finetuning/blob/main/Screen%20Recording%202026-01-02%20at%205.47.00%E2%80%AFPM.mov">
    <img src="https://img.shields.io/badge/▶%20Watch-Demo-blue?style=for-the-badge">
  </a>
</p>


---

## Overview

This repository contains code and files for fine-tuning a LLaMA-7B model using parameter-efficient techniques. The project demonstrates how to prepare data, configure training, run fine-tuning, and evaluate results. It includes a comprehensive Jupyter notebook with implementation details.

## Repository Structure

```
adapters/                      - Custom adapter modules for fine-tuning
configs/                       - Configuration files for training
data/processed/                - Processed dataset ready for training
src/                          - Source code for training and evaluation logic
ui/                           - User interface or related assets
fine-tuned-llama.ipynb        - Notebook with implementation walkthrough
requirements.txt              - Python dependencies
README.md                     - This documentation
```

## Requirements

Install the dependencies:

```bash
pip install -r requirements.txt
```

**System Requirements:**
- Access to the base LLaMA model weights (7B)
- GPU with sufficient VRAM (24GB+ recommended, or use quantization)
- Techniques such as 4-bit quantization with `bitsandbytes` and adapter-based training (LoRA/QLoRA) are used to reduce memory usage

## Data Preparation

Place your dataset files in the `data/processed/` directory. The dataset should be formatted as text instruction-response pairs suitable for supervised fine-tuning.

**Example expected format:**

```json
[
  {
    "instruction": "Explain the concept of transfer learning",
    "response": "Transfer learning involves taking a model trained on one task and adapting it to another related task by fine-tuning its weights..."
  }
]
```

Standard tokenization and preprocessing steps are performed in `src/data_processing.py` (or equivalent).

## Training

Use the training scripts in `src/` with appropriate configuration from `configs/`. The training process involves:

1. Loading the pretrained LLaMA-7B model
2. Freezing most parameters
3. Adding adapter layers (LoRA/QLoRA)
4. Training only the adapter parameters on the fine-tuning dataset

**Example command:**

```bash
python src/train.py \
  --config configs/finetune.yaml \
  --data_dir data/processed \
  --output_dir output
```

Model checkpoints will be saved to the specified output directory.

## Evaluation

After training, run the evaluation scripts to generate outputs from the fine-tuned model:

```bash
python src/evaluate.py \
  --model_dir output \
  --input examples/eval_prompts.json
```

Results can be compared to baseline generation outputs.

## Notebook Implementation

The file **[fine-tuned-llama.ipynb](fine-tuned-llama.ipynb)** contains a step-by-step implementation of the fine-tuning workflow, including:

- Data loading and preprocessing
- Model setup with adapter layers
- Training loop with logs
- Inference examples

Open this notebook in Jupyter to explore the full code and experiment with settings interactively.

## Notes

- This repository does not ship pretrained model weights. You need to download them separately (for example, via Hugging Face with appropriate licensing).
- Use GPU resources with sufficient memory (e.g., 24GB+ VRAM) or apply quantization and parameter-efficient fine-tuning to fit larger models on smaller hardware.
- Ensure you comply with LLaMA's licensing terms when using the model weights.

---

## License

Please refer to the LLaMA model license for usage terms and conditions.
