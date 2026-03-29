# SAGA: Scientific Agentic framework for Goal-oriented discovery with Adaptive objectives

SAGA is a generalist agentic framework for scientific discovery that automates the iterative process of objective design and hypothesis optimization. Rather than assuming a fixed set of objectives is known upfront, SAGA dynamically discovers and refines optimization objectives through a bi-level procedure: an **outer loop** that plans and evolves objectives, and an **inner loop** that optimizes candidate hypotheses against those objectives.

The framework comprises four core agentic modules:
- **Planner** decomposes the scientific goal into concrete, measurable objectives at each iteration
- **Implementer** converts proposed objectives into executable scoring functions
- **Optimizer** searches for candidate hypotheses that maximize the current objectives
- **Analyzer** evaluates optimization progress and provides actionable suggestions for the next iteration

SAGA supports three levels of human involvement:
- **co-pilot**: human collaborates with both planner and analyzer.
- **semi-pilot**: human reviews analyzer outputs only.
- **autopilot**: fully autonomous.

---

## Repository Structure

```
SAGA/
├── scileo_agent/           # Core SAGA framework
│
├── modules/                # Task-specific and shared module implementations
│   ├── shared/                     # Domain-agnostic implementations reusable across tasks
│   │   ├── scorer_creator/         # Implementer
│   │   ├── analyzer/               # Analyzer
│   │   ├── planner/                # Planner
│   │   ├── selector/               # Candidate selection utilities
│   │   ├── serializer/             # Candidate serialization/deserialization
│   │   └── knowledge_manager/      # Knowledge management
│   ├── dna_design/                 # DNA sequence design modules
│   └── small_molecule_drug_design/ # Small molecule drug design module
│
├── llm_configs/            # LLM model and credential configuration
│   ├── models.template.yaml        # Template for defining available LLM models
│   ├── claude_code.template.yaml   # Template for Claude Code model configuration
│   └── credentials.template.yaml   # Template for API keys and credentials
│
├── exps/                   # Experiment entry point scripts
│   ├── dna_design/                 # DNA design experiment
│   └── small_molecule_drug_design/ # Antibiotic design experiment
│
└── runs/                   # Run logs and results (auto-created at runtime)
```

---

## Usage

### 1. Hardware Requirements

SAGA requires a **Linux server** with one or more **GPUs** and **Docker** installed. Other platforms and configurations have not been thoroughly tested and are not guaranteed to work.

### 2. Environment Setup

**Create and activate a conda environment:**

```bash
conda create -n saga python=3.13
conda activate saga
```

**Install Python dependencies:**

```bash
pip install -r requirements.txt
```

**Install and start Docker:**

Ensure Docker is installed and the Docker daemon is running:

```bash
# Verify Docker is available
docker info
```

**Install Claude Code:**

Follow the official installation instructions at https://code.claude.com/docs/en/overview to install the Claude Code CLI.

**Pull required Docker images:**

SAGA uses Docker to run scoring functions and Claude Code. Pull the required images in advance to avoid long waits during experiments:

```bash
docker pull btyu24/scileo:v5-claude
docker pull btyu24/scileo:v4
docker pull btyu24/scileo:claude-agent-runner-251117
```

### 3. Configure LLM Models and Credentials

Copy all template config files to create your local configuration:

```bash
for f in llm_configs/*.template.yaml; do cp "$f" "${f/.template/}"; done
```

This creates three config files that you need to fill in:

- **`llm_configs/models.yaml`** — define which LLM models SAGA should use and how to call them
- **`llm_configs/claude_code.yaml`** — configure which Claude model is used by the Claude Code agent (in the Implementer and Analyzer)
- **`llm_configs/credentials.yaml`** — add your API keys for the providers you want to use (OpenAI, Anthropic, AWS Bedrock, etc.)

Open each file and follow the inline instructions to fill in your model settings and credentials.

### 4. Run Experiments

#### DNA Design

Open and run the experiment notebook:

```
exps/dna_design/exp_dna_design.ipynb
```

Launch Jupyter and execute all cells in the notebook. The experiment designs cell-type-specific enhancer sequences for the HepG2 cell line.

#### Antibiotic Design

Navigate to the experiment directory and run the script:

```bash
cd exps/small_molecule_drug_design
python exp_kp_drug.py --level <LEVEL>
```

The `--level` argument sets the autonomy level:

| Level | Mode | Description |
|-------|------|-------------|
| `1` | Co-pilot | Human collaborates with both the planner and analyzer at each iteration |
| `2` | Semi-pilot | Human reviews analyzer outputs and provides feedback; planner runs autonomously |
| `3` | Autopilot | All four modules run autonomously without human intervention |

For example, to run in autopilot mode:

```bash
python exp_kp_drug.py --level 3
```

Run logs and results will be saved automatically to the `runs/` directory.
