# CARDS Benchmark Framework

A comprehensive framework for benchmarking Large Language Models (LLMs) on climate misinformation detection and classification tasks, featuring the CARDS (Computer Assisted Recognition of Denial and Skepticism) dataset and hierarchical taxonomy.

## 🎯 Overview

The CARDS benchmark framework evaluates LLMs on their ability to detect and classify climate-related misinformation using a sophisticated hierarchical taxonomy of 103 categories across 7 major superclaim types. The framework supports both zero-shot and few-shot evaluation with dynamic example selection.

## 🚀 Features

- **Multi-Model Evaluation**: Compare 12+ LLM models across OpenAI, Anthropic, and fine-tuned variants
- **Hierarchical Classification**: 3-level taxonomy (superclaims → subclaims → subsubclaims)
- **ReCOT Generation**: Reverse Engineered Chain-of-Thought reasoning for enhanced few-shot learning
- **Dynamic Few-Shot**: Embedding-based similarity selection of relevant examples
- **Comprehensive Metrics**: Multi-label classification metrics with colored terminal display
- **Parallel Processing**: Efficient concurrent evaluation with rate limiting
- **Model Registry**: JSON-based model management with CLI tools
- **Provider Agnostic**: Unified interface for OpenAI and Anthropic APIs

## 📁 Project Structure

```
cards-2pO-paper/
├── 🎯 Core Framework
│   ├── main.py                 # Primary CLI entry point for benchmarks
│   ├── benchmark.py            # CARDSBenchmark class - orchestrates evaluation
│   ├── models.py               # ModelClient & ResponseParser - LLM integration
│   ├── config.py               # Configuration management & environment setup
│   ├── embeddings.py           # EmbeddingManager - few-shot example selection
│   ├── metrics.py              # Metrics calculation & colored display
│   └── prompts.py              # System prompts & classification taxonomy
│
├── 🤖 Model Management
│   ├── model_registry.py       # CLI for model registration/management
│   └── models.json             # Model definitions (12 pre-configured models)
│
├── 🧠 ReCOT Module
│   ├── recot/
│   │   ├── __init__.py
│   │   ├── core.py             # ReCOTGenerator - reasoning chain generation
│   │   └── cli.py              # CLI interface for ReCOT generation
│
├── 📊 Data Directory
│   ├── data/
│   │   ├── 📋 Core Datasets
│   │   │   ├── congress_test.csv      # Main benchmark dataset (congressional texts)
│   │   │   ├── fewshot.csv            # Original few-shot examples
│   │   │   ├── recot_fewshot.csv      # ReCOT-enhanced few-shot examples
│   │   │   ├── taxonomy.csv           # Complete classification taxonomy
│   │   │   └── recot_error_analysis.csv # Error analysis exclusion list
│   │   │
│   │   ├── 🗺️ Mapping Files
│   │   │   └── mapping/
│   │   │       ├── final_claims_dict.json  # Text → final expert labels
│   │   │       └── true_claims_dict.json   # Text → ground truth labels
│   │   │
│   │   ├── 📈 Results
│   │   │   └── results/
│   │   │       ├── fewshot_results.csv     # Few-shot evaluation results
│   │   │       └── zeroshot_results.csv    # Zero-shot evaluation results
│   │   │
│   │   └── 🔬 Research Data
│   │       └── evals/paper/                # Paper experiment results
│   │           ├── cards_*_predictions.parquet
│   │           ├── claude_*_predictions.parquet
│   │           └── fewshot_results_metrics.parquet
│
└── 📋 Configuration
    ├── requirements.txt            # Python dependencies
    └── .env.example               # Environment template
```

## 🏗️ Architecture Overview

### Core Components

#### 1. **CARDSBenchmark** (`benchmark.py`)

- **Purpose**: Orchestrates multi-model evaluation pipeline
- **Key Features**:
  - Parallel processing with ThreadPoolExecutor
  - Dynamic few-shot example selection via embeddings
  - Structured output parsing with fallback mechanisms
  - Rate limiting for different API providers
- **Data Flow**: Input text → Embedding (if few-shot) → Prompt construction → LLM API → Response parsing → Metrics

#### 2. **Model Management** (`models.py`, `model_registry.py`, `models.json`)

- **ModelClient**: Unified interface for OpenAI and Anthropic APIs with retry logic
- **ResponseParser**: Extracts structured classifications from LLM responses
- **Model Registry**: 12 pre-configured models:
  - **OpenAI Base**: GPT-4o, GPT-4o-Mini, GPT-4o-Latest
  - **Anthropic Base**: Claude-3.5, Claude-3.7, Claude-4-Sonnet
  - **CARDS Fine-tuned**: GPT models trained on CARDS data
  - **CARDS Nano**: Specialized smaller models

#### 3. **ReCOT Generation** (`recot/core.py`)

- **Purpose**: Generate Reverse Engineered Chain-of-Thought reasoning
- **Process**: Text + True Labels → Detailed reasoning chains → Enhanced training data
- **Output**: Individual JSON files + combined parquet for analysis

#### 4. **Evaluation & Metrics** (`metrics.py`)

- **Multi-label Metrics**: F1, Precision, Recall (Micro/Macro/Samples)
- **Additional Metrics**: Accuracy, Hamming Loss, Matthews Correlation
- **Hierarchical Processing**: Claims processed at 3 levels
- **Colored Display**: Terminal-based results with performance color coding

## 🔧 Installation & Setup

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/project-c3ds/cards-2pO-paper.git
cd cards-2pO-paper

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. API Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your API keys
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
```

### 3. Verify Installation

```bash
# List available models
python main.py --list_models

# Test claim processing
python metrics.py --test-claims
```

## 🚀 Usage Examples

### Basic Evaluation

```bash
# Run complete evaluation with colored metrics
python metrics.py

# Compare specific models on default dataset
python main.py --models 1 3 5

# Quick test with sample data
python main.py --sample --sample_size 10
```

### Few-Shot Evaluation

```bash
# Load few-shot examples and run evaluation
python main.py --use_fewshot --fewshot_data_path data/recot_fewshot.csv --models 1

# Run few-shot with original examples
python main.py --use_fewshot --fewshot_data_path data/fewshot.csv --models 3
```

### Custom Data

```bash
# Evaluate on your own dataset
python main.py --input_file your_data.csv --text_column "content" --models 2 4

# Save results to specific location
python main.py --output_file results/my_evaluation.csv --models 1 2 3
```

### ReCOT Generation

```bash
# Generate reasoning chains for few-shot examples
python recot/cli.py --input_file data/fewshot.csv --sample 20

# Use specific models for ReCOT generation
python recot/cli.py --input_file data/fewshot.csv --models 1 3 --max_workers 3

# Generate for full dataset
python recot/cli.py --input_file data/fewshot.csv --models 1
```

### Model Management

```bash
# List all registered models
python model_registry.py list

# Add new model
python model_registry.py register "GPT-4-Turbo" openai "gpt-4-turbo-2024-04-09"

# Update existing model
python model_registry.py update 12 --temperature 0.2 --max_tokens 2000

# Remove model
python model_registry.py remove 12
```

## 📊 Data Files Reference

### Core Datasets

- **`data/congress_test.csv`**: Congressional texts on climate topics with expert annotations
- **`data/fewshot.csv`**: Curated examples for few-shot learning
- **`data/recot_fewshot.csv`**: Enhanced few-shot examples with generated reasoning chains
- **`data/taxonomy.csv`**: Classification taxonomy categories with descriptions

### Mapping & Configuration

- **`data/mapping/final_claims_dict.json`**: Maps texts to expert-annotated final labels
- **`data/mapping/true_claims_dict.json`**: Maps texts to ground truth labels for evaluation
- **`models.json`**: Model registry with 12 pre-configured LLM models

### Results & Analysis

- **`data/results/fewshot_results.csv`**: Few-shot evaluation results across all models
- **`data/results/zeroshot_results.csv`**: Zero-shot evaluation results across all models
- **`data/recot_error_analysis.csv`**: Texts excluded from evaluation due to errors

## 🎯 Key Metrics & Evaluation

The framework computes comprehensive multi-label classification metrics:

### Primary Metrics

- **Samples F1**: F1 score computed per sample then averaged
- **Samples Precision**: Precision computed per sample then averaged
- **Samples Recall**: Recall computed per sample then averaged
- **Accuracy**: Exact match ratio for multi-label predictions

### Additional Metrics

- **Micro/Macro F1**: Alternative F1 averaging strategies
- **Hamming Loss**: Fraction of incorrect labels
- **Matthews Correlation**: Correlation between predictions and ground truth

### Performance Display

Results are displayed with color-coded performance:

when you run the `metrics.py` script, the results are displayed with color-coded performance:

- 🔴 **High Performance** (≥0.8): Red highlighting
- 🟡 **Medium Performance** (0.6-0.8): Yellow highlighting
- 🔵 **Lower Performance** (<0.6): Cyan highlighting

### Benchmark Results

Performance comparison across different model categories on the CARDS dataset:

| **Model**                    | **F1**    | **Precision** | **Recall** | **Accuracy** | **Hamming Loss** | **MCC**   |
| ---------------------------- | --------- | ------------- | ---------- | ------------ | ---------------- | --------- |
| **Zero-shot Learning**       |           |               |            |              |                  |           |
| Claude-Sonnet-3.5            | 0.834     | 0.838         | 0.836      | 0.789        | 0.006            | 0.800     |
| Claude-Sonnet-3.7            | **0.881** | **0.890**     | **0.879**  | **0.836**    | **0.005**        | **0.846** |
| GPT-4o                       | 0.671     | 0.681         | 0.670      | 0.628        | 0.010            | 0.637     |
| GPT-4o-Mini                  | 0.506     | 0.518         | 0.503      | 0.470        | 0.012            | 0.515     |
| **Few-shot Learning**        |           |               |            |              |                  |           |
| Claude-Sonnet-3.5            | 0.820     | 0.825         | 0.823      | 0.765        | 0.006            | 0.782     |
| GPT-4o                       | 0.633     | 0.637         | 0.637      | 0.582        | 0.012            | 0.592     |
| GPT-4o-Mini                  | 0.550     | 0.558         | 0.554      | 0.495        | 0.014            | 0.501     |
| **Fine-tuned Models**        |           |               |            |              |                  |           |
| CARDS-mini-GPT               | 0.815     | 0.828         | 0.811      | 0.772        | 0.006            | 0.773     |
| CARDS-mini-GPT-2024-12-05    | 0.840     | 0.855         | 0.836      | 0.797        | 0.006            | 0.801     |
| CARDS-mini-Sonnet            | 0.825     | 0.838         | 0.821      | 0.787        | 0.006            | 0.784     |
| CARDS-mini-Sonnet-2024-12-05 | 0.852     | 0.866         | 0.848      | 0.809        | **0.005**        | 0.809     |

**Key Findings:**

- **Best Overall**: Claude-Sonnet-3.7 (zero-shot) achieves highest F1 score of 0.881
- **Fine-tuning Impact**: CARDS-tuned models show significant improvement over base models
- **Claude vs GPT**: Claude models consistently outperform GPT models across all metrics
- **Few-shot Effect**: Mixed results - some models benefit while others show slight degradation

## 🔬 Research Applications

### Academic Use Cases

1. **LLM Comparison**: Benchmark different models on climate misinformation detection
2. **Few-Shot Learning**: Study impact of example selection on performance
3. **Reasoning Analysis**: Analyze chain-of-thought generation effectiveness
4. **Fine-Tuning**: Create training data for domain-specific model adaptation

### Extensibility

- **Custom Taxonomies**: Adapt classification system for other domains
- **New Models**: Easy integration of additional LLM providers
- **Enhanced Metrics**: Extend evaluation with domain-specific measures
- **Data Augmentation**: Use ReCOT for generating synthetic training examples

## 🤝 Contributing

1. **Model Addition**: Add new models via `model_registry.py`
2. **Taxonomy Extension**: Modify `prompts.py` for new categories
3. **Metric Enhancement**: Extend `metrics.py` with additional evaluation measures
4. **Data Processing**: Add new preprocessing functions to `benchmark.py`

## 🌐 Live Demo & API

The CARDS classification system is available as an interactive chatbot and API:

- **Website**: https://www.discourselab.ai/
- **Interactive Demo**: Test climate misinformation detection in real-time
- **API Access**: Integrate CARDS classification into your applications

## 📝 Citation

If you use the CARDS framework in your research, please cite:

```bibtex
@article{cards2pO2025,
  title={Decoding Delay: Three Decades of Climate Change Opposition in the United States Congress},
  author={Travis G. Coan and Ranadheer Malla and Mirjam O. Nanko and William Kattrup and J. Timmons Roberts and John Cook and Constantine Boussalis},
  journal={[Journal]},
  year={2025}
}
```

## 👥 Authors

**Travis G. Coan**¹ᐟ\*, **Ranadheer Malla**¹, **Mirjam O. Nanko**¹, **William Kattrup**², **J. Timmons Roberts**², **John Cook**³, **Constantine Boussalis**⁴

¹ University of Exeter, Centre for Climate Communication and Data Science, Exeter, EX4 4PE, UK  
² Brown University, Climate and Development Lab, Providence, RI 02912, USA  
³ University of Melbourne, Melbourne Centre for Behaviour Change, Parkville, 3010, Australia  
⁴ Trinity College Dublin, Department of Political Science, Dublin 2, Ireland

\*Corresponding author

## 📄 License

See [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Issues**: Report bugs via GitHub Issues
- **Documentation**: Check individual module docstrings
- **Examples**: Use the CLI commands and examples provided in this README
