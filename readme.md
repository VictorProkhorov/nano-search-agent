# A Simple Search-Agent (**Under Development**)

## (SFT) Fine-Tuning Pipeline
 
A pipeline for synthesizing, filtering, and fine-tuning a Qwen2.5-Instruct model to perform multi-turn tool-use (web search / Wikipedia retrieval) reasoning on HotpotQA-style question answering, following a Search-R1 / R1-Searcher-style `<think>/<tool_call>/<tool_response>/<answer>` format.
 
## Pipeline Overview
 
1. **Trajectory synthesis** (`data/synthesis/trajectory_collection.py`) — runs a base model over sampled HotpotQA questions, letting it call a search/retriever tool across multiple turns, and saves the resulting message trajectories.
2. **Trajectory filtering** (`data/post_processing/trajectories.py`) — validates trajectory format (correct tag structure/ordering) and filters to only those with answers matching the gold answer, saving the result as a HuggingFace `Dataset`.
3. **SFT training** (`train.py`) — LoRA fine-tunes Qwen2.5-0.5B-Instruct on the filtered trajectories, with custom loss masking that ignores prompt tokens and `<tool_response>` blocks.
4. **Evaluation** (`evals/compare_answers.py`) — generates trajectories with the fine-tuned model and computes exact-match accuracy against gold answers.
## Project Structure
 
```
src/
├── tools.py                          # Search/retrieval tools and schemas
├── train.py                          # LoRA SFT training script
├── data/
│   ├── synthesis/
│   │   └── trajectory_collection.py  # Generate trajectories via tool-use rollouts
│   └── post_processing/
│       └── trajectories.py           # Validate, filter, and save trajectories
└── evals/
    └── compare_answers.py            # Exact-match evaluation
```
