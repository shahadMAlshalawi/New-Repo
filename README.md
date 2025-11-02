# Modular Arabic Visual Question Answering System using Pre-Trained Models

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shahadMAlshalawi/Modular-Arabic-VQA/blob/main/notebooks/OKVQA_ar_gemini_experiments_Final.ipynb)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shahadMAlshalawi/Modular-Arabic-VQA/blob/main/notebooks/VQAv2_ar_gemini_experiments_Final.ipynb)


<p align="center">
  <img src="assets/images/Examples_OKVQA-ar.jpg" alt="VQA Examples" width="800"><br>
    <em>Figure 3. Representative examples comparing the performance of Gemini 1.5 Flash model on Arabic VQA on the OKVQA-ar validation set. Upper part shows its predictions without context captions, and bottom part shows its predictions when introduced with context captions under Modular Arabic VQA system configuration. Right side presents failure cases for both configurations.</em>
</p>

<p align="center">
  <img src="assets/images/Examples_VQAv2-ar.jpg" alt="VQA Examples" width="800"><br>
    <em>Figure 4. Representative examples comparing the performance of Gemini 1.5 Flash model on Arabic VQA on the VQAv2-ar validation set. Upper part shows its predictions without context captions, and bot-tom part shows its predictions when introduced with context captions under Modular Arabic VQA system configuration. Right side presents failure cases for both configurations.</em>
</p>

## 🧠 Overview

**Modular Arabic Visual Question Answering System (Modular Arabic VQA)** is a **language-mediated framework** for answering open-ended questions about images in **Arabic**.
Unlike conventional vision–language models that rely on dense visual embeddings, **Modular Arabic VQA** represents images entirely through **textual descriptions**, enabling efficient and interpretable reasoning via large language models (LLMs).

This framework integrates multiple Arabic and multilingual pretrained models for image captioning, and natural-language processing, forming a modular and extensible pipeline for experimentation and research in Arabic VQA field.

## Key Features:
-  🧩 **Modularity:** Modular framework allowing easy integration or replacement of captioning, language, and many more components.
-  💡 **Lightweight & Interpretable:** Eliminates multimodal retraining by relying on textual representations, reducing computational cost while maintaining robust reasoning performance.
-  🌍 **Arabic Support:** Designed to handle Arabic datasets and language processing tasks.
-  🔬 **Flexible Experimentation:** Supports multiple captioning models (AraBERT32-Flickr8k, Violet, GPT-4o), similarity-based caption-selection strategies, and diverse evaluation metrics ([BLEU](https://huggingface.co/spaces/evaluate-metric/bleu), [BERTScore](https://arxiv.org/abs/1904.09675), [Fuzz Accuracy](https://github.com/mbzuai-oryx/Camel-Bench/blob/main/scripts/fuzz_eval.py)).
-  📈 **Research-Ready Pipeline:** Facilitates ablation studies, benchmarking, and reproducible evaluations of modular VQA configurations..



## Installation

#### Option 1: Install via `pip`
```bash
pip install git+https://github.com/shahadMAlshalawi/Modular-Arabic-VQA.git --quiet
```
#### Option 2: Clone Repository and Install in Editable Mode
```bash
git clone https://github.com/shahadMAlshalawi/Modular-Arabic-VQA.git
cd Modular-Arabic-VQA
pip install -e .
```
### Option 3: Use Conda Environment
```bash 
git clone https://github.com/shahadMAlshalawi/Modular-Arabic-VQA.git
cd Modular-Arabic-VQA
conda env create -f environment.yml
conda activate aravqa
pip install -e .
```

## Architecture

The Modular Arabic VQA framework is structured into multiple reusable modules, allowing customization and extensibility:

- **Core:** Contains shared utilities and configurations.

- **Datasets:** Manages dataset loading, processing, and preparation.

- **Modules:** Implements various models for captioning and question answering.


## Quick Start


```python 
# Import necessary modules
from aravqa.core.config import Config
from datasets import load_dataset
from aravqa.datasets.utils import prepare_dataset, compute_similarity_captions, compute_bleu_score
from aravqa.datasets import OKVQADataset, OKVQADataLoader
from aravqa.modules.captioning import BiTCaptioner, VioletCaptioner
from aravqa.modules.question_answering import GeminiAnswerer
from aravqa.modules.evaluation import BLEUEvaluator, AccuracyEvaluator
import pandas as pd
import textwrap

# Initialize configuration
config = Config()
config.API_KEY = "your_google_generative_ai_api_key"
config.MODEL_NAME = "models/gemini-1.5-flash"
config.GENERATION_CONFIG = {
    "temperature": 0.0,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 20,
    "response_mime_type": "text/plain",
}
config.SYSTEM_INSTRUCTION = textwrap.dedent("""
    You are a highly capable language model specialized in answering questions based on provided image captions.
    Your task is to analyze the captions and generate accurate, concise answers in the same language as the question.
    Ensure your response is relevant, clear, and avoids unnecessary details.
""").strip()

config.PROMPT_TEMPLATE = textwrap.dedent("""
    Analyze the following image captions and answer the given question in the same language:
    Captions: {context}
    Question: {question}
    Answer concisely:
""").strip()

# Load datasets
bds = load_dataset(config.BDS_PATH, split=config.SPLIT)
vds = load_dataset(config.VDS_PATH, split=config.SPLIT)
gpt4oDS = load_dataset(config.GPT4oDS_PATH,split=Config.SPLIT)

# Prepare datasets
BDS = prepare_dataset(bds, language=config.LANGUAGE)
VDS = prepare_dataset(vds, language=config.LANGUAGE)
GPT4oDS = prepare_dataset(gpt4oDS, language=config.LANGUAGE)

# Compute similarities for captions
BDS = compute_similarity_captions(BDS, question_similarity_scorer=compute_bertscore, answer_similarity_scorer=compute_bertscore)
VDS = compute_similarity_captions(VDS, question_similarity_scorer=compute_bertscore, answer_similarity_scorer=compute_bertscore)
GPT4oDS = compute_similarity_captions(GPT4oDS, question_similarity_scorer=compute_bertscore, answer_similarity_scorer=compute_bertscore)

# Combine datasets
dataset = OKVQADataset(BDS, VDS, GPT4oDS)

# Initialize DataLoader
dataloader = OKVQADataLoader(dataset, config).get_dataloader()

# Initialize LLM Model
llm = GeminiAnswerer(config)

# Generate answers
outputs = llm.generate_from_dataloader(dataloader)
outputs_df = pd.DataFrame.from_dict(outputs)

# Evaluate results
bleu_evaluator = BLEUEvaluator(max_order=4)
bertScore_evaluator = BERTScoreEvaluator()
fuzzy_evaluator = FuzzEvaluator(OPENAI_API_KEY)


bleu_results = bleu_evaluator.evaluate(predictions=outputs["predictions"], references= outputs["answers"])
bertScore_results = bertScore_evaluator.evaluate(predictions=outputs['predictions'], references= outputs['answers'])
fuzzy_results = fuzzy_evaluator.evaluate(predictions=outputs['predictions'], references=outputs['answers'], questions=outputs['questions'])

# Display results
print("BLEU Results:", bleu_results)
print("BertScore Results:", bertScore_results)
print("Fuzz Accuracy Results:", fuzzy_results)



```


## Citation

If you use Modular Arabic VQA in your research, please cite the repository:
```
@misc{Modular Arabic VQA,
  author = {Shahad Alshalawi},
  title = {Modular Arabic Visual Question Answering System},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/shahadMAlshalawi/Modular-Arabic-VQA}},
}
```




