# A Nano Search-Agent (Under Development)

## Overview
 
A ReAct based search agent. I conduct experiments on NVIDIA GeForce RTX 5060 Ti, CUDA:13.0

---
 
## Project Structure
 
```
.
├── src/
│   ├── tools.py                           # MCP server with DuckDuckGo search tool
│   ├── train.py                           # LoRA fine-tuning pipeline
│   ├── mcp_tool_converter.py              # Converts MCP schemas to LLM format
│   ├── conf/
│   │   └── train.yaml                     # Training hyperparameters
│   ├── data/
│   │   ├── synthesis/
│   │   │   ├── trajectory_collection.py   # Agent loop for trajectory generation
│   │   │   ├── prompts.py                 # Search agent system prompt
│   │   │   └── conf/
│   │   │       ├── train_traj.yaml        # Config for training trajectory collection
│   │   │       └── eval_traj.yaml         # Config for eval trajectory collection
│   │   ├── post_processing/
│   │   │   ├── filter_trajectories.py     # Trajectory format validation & filtering
│   │   │   └── conf/
│   │   │       ├── train_traj_filter.yaml # Config for trajectory filtering
│   └── evals/
│       ├── answer_metrics.py              # Answer correctness evaluation
│       ├── trajectory_metrics.py          # Trajectory quality metrics
│       ├── no_tool_baseline.py            # Baseline without tool use
│       ├── evaluate.py                    # Main evaluation orchestrator
│       └── conf/
│           └── eval.yaml                  # Evaluation config
├
└── README.md

```
---

## Usage
Make sure you are outside /src and mkdir /data and mkdir /models outside the /src
```
.
├── src/
│      
│       
├── data/
│   ├── trajectories/
│   │   └── filtered
│   
├── models/
│   └── Qwen2.5-0.5B-Instruct-Search-LoRA/   # Trained model

```


### Trajectory Collection (For Training)

```
python3 -m src.data.synthesis.trajectory_collection --config-name train_traj
```

### Trajectory Filtering

```
python3 -m src.data.post_processing.filter_trajectories --config-name=train_traj_filter
```

### Train

```
python3 -m src.train --config-name=train
```

### Trajectory Collection (For Eval)

```
python3 -m src.data.synthesis.trajectory_collection --config-name eval_traj
```

### Evaluation
 
```
python3 -m src.evals.evaluate --config-name eval
```
 
**Output:**
```
╔════════════════════════════════════════════╗
║             EVALUATION SUMMARY             ║
╚════════════════════════════════════════════╝
 
ANSWER
    Token F1:           0.485
    Exact Match (EM):   0.412
    Case-Insensitive:   0.441
    Semantic Sim (SAS): 0.652
    Semantic Sim (Judge): 0.723
    Attribution:        0.581
 
TRAJECTORY
    Answer Extraction Rate:     0.945
    % with Reasoning:           0.887
    Av. Reasoning Steps:        2.134
    % with Tool Calls:          0.923
    Av. Tool Calls:             2.456
    Tool Error Rate:            0.089
    Av. Conversation Turns:     5.234
```
 
---
