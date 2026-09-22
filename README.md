# Bangla-English Code-Mixed Sentiment Analysis

An end-to-end natural language processing system for classifying Bangla-English
code-mixed text into four sentiment categories: **Positive**, **Negative**,
**Neutral**, and **Mixed**.

The project covers the complete NLP workflow, including corpus preparation,
text normalization, feature extraction, word embeddings, sequence modelling,
self-attention, comparative evaluation, and interactive inference. A
validation-tuned ensemble combines complementary NLP models to produce one
final prediction.

## Key features

- Normalizes noisy Bangla-English user-generated text.
- Preserves both Bangla and English Unicode characters during preprocessing.
- Uses word and character TF-IDF alongside embedding, recurrent, and attention-based models.
- Combines four trained models into one final ensemble prediction.
- Reports accuracy, macro precision, macro recall, and macro F1.
- Includes confusion matrices and error analysis for model interpretation.
- Provides both a Streamlit web application and command-line interface.
- Organizes each experimental stage in a corresponding Jupyter notebook.

## System architecture

```text
Input text
    │
    ▼
Normalization and tokenization
    │
    ├── Word + character TF-IDF + Logistic Regression ────┐
    ├── TF-IDF-weighted Word2Vec + Logistic Regression ───┤
    ├── Bidirectional LSTM ────────────────────────────────┤
    └── Transformer Encoder ───────────────────────────────┤
                                                            ▼
                             Validation-tuned probability averaging
                                                            │
                                                            ▼
                            Final sentiment and class probabilities
```

Every component processes the same input. We tested weights in 5% steps on the
validation split, requiring every model to contribute at least 5%. The selected
weights are **85% word-and-character TF-IDF** and **5% each** for Word2Vec,
BiLSTM, and Transformer. Test-set results were not used to choose the weights.
Character fragments help recognize spelling variations such as `valo`, `bhalo`,
and `vhalo` without a hand-written spelling dictionary.

For a sentence containing `but`, `kintu`, or `tobe`, a conservative contrast
rule can prioritize a clearly stronger negative clause after that word. It
applies only if the whole sentence was predicted Mixed, the first clause has at
least 50% Positive probability, and the later clause has at least 70% Negative
probability and exceeds the first clause's Positive probability by 15 points.
The clause probabilities come from the word-and-character TF-IDF model. This
rule makes `chele valo kintu ektu bar e jay mod khay` Negative. It changed no
labels in the saved validation or test splits, so that example is the direct
evidence for this application-specific behavior. The corpus often labels text
with both positive and negative phrases as Mixed.

### Simple explanation for a class presentation

1. Clean the sentence while keeping Bangla and English letters.
2. Turn words, word pairs, and short character fragments into TF-IDF numbers.
   Character fragments help when the same Bangla word is typed in different ways.
3. Four trained models estimate probabilities for Positive, Negative, Neutral,
   and Mixed. Average them with weights selected using validation data.
4. If a strong negative clause follows `kintu`, let that clause decide a Mixed
   result only when it is clearly stronger than the earlier positive clause.

## Dataset

This project uses **BnSentMix: A Diverse Bengali-English Code-Mixed Dataset for
Sentiment Analysis**, developed by Sadia Alam, Md Farhan Ishmam, Navid Hasin
Alvee, Md Shahnewaz Siddique, Md Azam Hossain, and Abu Raihan Mostofa Kamal.

- [Official BnSentMix repository](https://github.com/Nishita2000/BnSentMix)
- [Research paper in the ACL Anthology](https://aclanthology.org/2025.loreslm-1.4/)
- Local source file: `data/raw/dataset.csv`

The original corpus contains user-generated text collected from YouTube
comments, Facebook comments, and e-commerce reviews. The local dataset contains
20,015 rows. After removing missing, empty, and duplicate records, 19,822 valid
samples remain.

| Numeric label | Sentiment |
| ---: | --- |
| 0 | Positive |
| 1 | Negative |
| 2 | Neutral |
| 3 | Mixed |

Please consult the original dataset repository and paper for collection details,
licensing information, and citation requirements.

## Selected NLP models

The system contains only the four models used by the final ensemble:

| Component | Purpose |
| --- | --- |
| Word + character TF-IDF + Logistic Regression | Captures words, short phrases, and character fragments for spelling variation |
| TF-IDF-weighted Word2Vec + Logistic Regression | Combines semantic token vectors with corpus-level term importance |
| Bidirectional LSTM | Reads token sequences in both directions to model word order |
| Transformer Encoder | Uses position-aware self-attention to model contextual relationships |

### Selection from the NLP lab material

The project uses the lab techniques that directly support code-mixed sentiment
classification. Other lab exercises are intentionally excluded instead of being
added without a task-specific reason.

| Lab material | Use in this project | Decision |
| --- | --- | --- |
| Regex cleaning and tokenization | Unicode-safe normalization that preserves Bangla, Romanized Bangla, and negation words | Core preprocessing |
| TF-IDF | Sparse word unigram/bigram and character 3–5 gram features | Used by the ensemble |
| Logistic Regression | Four-class probability prediction from TF-IDF and weighted Word2Vec vectors | Used by the ensemble |
| Word2Vec | TF-IDF-weighted semantic document representation | Used by the ensemble |
| Bidirectional LSTM | Forward and backward sequence modelling | Used by the ensemble |
| Transformer encoder | Position-aware self-attention classifier | Used by the ensemble |

English-only stemming, lemmatization, stop-word removal, and automatic spelling
correction are not applied. They can remove sentiment-bearing words or alter
Romanized Bangla incorrectly. N-gram language generation, Shannon's guessing
game, POS tagging, autoregressive text generation, and Seq2Seq translation are
also outside the scope of sentiment classification.

## Evaluation summary

The final ensemble is recorded as experiment `M6.1` in
`results/metrics.csv`.

| Metric | Test score |
| --- | ---: |
| Accuracy | 0.7408 |
| Macro precision | 0.7457 |
| Macro recall | 0.6967 |
| Macro F1 | 0.7143 |

Macro-averaged metrics are emphasized because they give equal importance to all
four sentiment classes, regardless of class frequency. In the prior local run,
the ensemble scored 0.7209 accuracy and 0.6983 macro F1 on the same test split.

## Repository structure

```text
Bangla-English-Code-Mixed-Sentiment-Analysis/
├── data/
│   ├── raw/
│   │   └── dataset.csv
│   ├── processed/
│   │   └── processed_sentiment.csv
│   └── splits/
│       ├── train.csv
│       ├── validation.csv
│       └── test.csv
├── models/                    generated model artifacts
├── notebooks/
│   ├── 01_preprocessing.ipynb
│   ├── 02_classical_models.ipynb
│   ├── 03_word_embeddings.ipynb
│   ├── 04_bilstm.ipynb
│   ├── 05_transformer.ipynb
│   └── 06_final_ensemble.ipynb
├── results/
│   ├── confusion_matrices/
│   ├── error_analysis.csv
│   └── metrics.csv
├── src/
│   ├── config.py
│   ├── dataset_utils.py
│   ├── evaluation.py
│   ├── neural_models.py
│   ├── prediction.py
│   ├── preprocessing.py
│   ├── train_neural_models.py
│   └── word_embeddings.py
├── app.py                     Streamlit web application
├── check_environment.py       lightweight dependency check used by run.ps1
├── run.ps1                    one-command dependency check and web launcher
├── run_project.py             command-line entry point
├── setup.ps1                  Windows environment setup
├── requirements.txt
└── README.md
```

The main project workflow is visible in the notebooks. The `src/` directory
contains only code reused by multiple notebooks or required by the web and
command-line interfaces.

## Requirements

- Windows PowerShell for the provided automatic setup script
- Python 3.11 or 3.12
- `uv` for automatic environment creation

Python 3.14 is not recommended for this project because a compatible prebuilt
Gensim wheel may not be available on Windows.

## Installation

Clone the repository and enter the project directory:

```powershell
git clone https://github.com/abdullahshahporan/Bangla-English-Code-Mixed-Sentiment-Analysis.git
cd Bangla-English-Code-Mixed-Sentiment-Analysis
```

Create a Python 3.12 virtual environment and install the dependencies:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

The script creates a local `.venv` directory. Virtual environments, caches,
private `.env` files, and generated model binaries are excluded from Git.

## Generate the model artifacts

Pretrained binary artifacts are not stored in the repository. Execute the six
notebooks in numeric order from JupyterLab, or run them from PowerShell:

```powershell
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks\01_preprocessing.ipynb
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks\02_classical_models.ipynb
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks\03_word_embeddings.ipynb
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks\04_bilstm.ipynb
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks\05_transformer.ipynb
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks\06_final_ensemble.ipynb
```

These commands generate the four artifacts required by the ensemble:

```text
models/tfidf_logistic.pkl
models/word2vec_sentiment.pkl
models/bilstm_best.pt
models/transformer_best.pt
```

## Run the web application

After the model artifacts are available, run this single command from the
project directory:

```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1
```

The launcher always uses the project's Python 3.12 `.venv`, checks the required
packages and model files, installs missing packages when possible, and then
starts Streamlit. This avoids accidentally using a different Python environment
that does not contain packages such as Gensim.

Open the URL printed in the terminal, normally
`http://localhost:8501`. Enter a Bangla-English sentence and select
**Analyze sentiment**. The application displays:

1. The original sentence.
2. The normalized text and tokens.
3. One ensemble sentiment prediction.
4. The probability assigned to each sentiment class.

Example input:

```text
camera bhalo but battery kharap
```

The interface intentionally has no model selector because the four deployed
models jointly produce the final result.

## Command-line prediction

The same ensemble can be used without the web interface:

```powershell
.\.venv\Scripts\python.exe run_project.py "service ta valo na"
```

Example output:

```text
Model: Validation-tuned NLP ensemble
Predicted sentiment: Negative
Probabilities:
  Positive: ...
  Negative: ...
  Neutral : ...
  Mixed   : ...
```

## Notebooks

The notebooks contain the main project stages in executable order:

1. Text preprocessing and corpus analysis
2. Word-and-character TF-IDF Logistic Regression
3. TF-IDF-weighted Word2Vec Logistic Regression
4. Bidirectional LSTM
5. Transformer encoder
6. Final ensemble evaluation and error analysis

Reusable functions and model classes remain in `src/`, preventing repeated code
between notebooks 4 and 5 and ensuring that the saved models can be loaded by
the application. Fixed files in `data/splits/` give all four components the same
reproducible training, validation, and test partitions.

Start JupyterLab with:

```powershell
.\.venv\Scripts\python.exe -m jupyter lab
```

## Outputs

- `results/metrics.csv` contains the four component results and final ensemble result.
- `results/confusion_matrices/` contains class-level evaluation figures.
- `results/error_analysis.csv` contains incorrectly classified test examples for
  qualitative analysis.
- `models/` contains locally generated inference artifacts.

## Scope and limitations

This system is designed for the language patterns represented in BnSentMix.
Performance may decrease for other domains, dialects, spelling conventions, or
sentiment definitions. Predictions should therefore be interpreted as model
estimates rather than definitive judgments about a writer's intent.
