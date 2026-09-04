# Bangla-English Code-Mixed Sentiment Analysis

An end-to-end natural language processing system for classifying Bangla-English
code-mixed text into four sentiment categories: **Positive**, **Negative**,
**Neutral**, and **Mixed**.

The project covers the complete NLP workflow, including corpus preparation,
text normalization, feature extraction, word embeddings, sequence modelling,
self-attention, comparative evaluation, and interactive inference. A
validation-weighted ensemble combines complementary NLP models to produce one
final prediction.

## Key features

- Normalizes noisy Bangla-English user-generated text.
- Preserves both Bangla and English Unicode characters during preprocessing.
- Compares classical, embedding-based, recurrent, and attention-based methods.
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
    ├── TF-IDF + Logistic Regression ─────────────────────┐
    ├── TF-IDF-weighted Word2Vec + Logistic Regression ───┤
    ├── Bidirectional LSTM ────────────────────────────────┤
    └── Transformer Encoder ───────────────────────────────┤
                                                            ▼
                             Validation-weighted probability averaging
                                                            │
                                                            ▼
                            Final sentiment and class probabilities
```

Every component processes the same input. Its validation macro F1 score is used
as its ensemble weight, allowing stronger validation models to contribute more
to the final probability distribution. Test-set results are not used to select
or weight the models.

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

## NLP methods

The experiments are grouped into four modelling families:

| Family | Models and representations |
| --- | --- |
| Classical NLP | Bag-of-Words and TF-IDF with Naive Bayes or Logistic Regression |
| Word embeddings | Mean Word2Vec and TF-IDF-weighted Word2Vec with Logistic Regression |
| Recurrent networks | Vanilla RNN and Bidirectional LSTM with trainable embeddings |
| Self-attention | Transformer encoder with positional encoding |

The deployed ensemble uses the strongest practical artifacts from these
experiments: TF-IDF Logistic Regression, TF-IDF-weighted Word2Vec, Bidirectional
LSTM, and Transformer Encoder.

## Evaluation summary

The final ensemble is recorded as experiment `M6.1` in
`results/metrics.csv`.

| Metric | Test score |
| --- | ---: |
| Accuracy | 0.7192 |
| Macro precision | 0.7110 |
| Macro recall | 0.6868 |
| Macro F1 | 0.6970 |

Macro-averaged metrics are emphasized because they give equal importance to all
four sentiment classes, regardless of class frequency.

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
│   ├── 04_rnn_bilstm.ipynb
│   ├── 05_transformer.ipynb
│   └── 06_final_comparison.ipynb
├── results/
│   ├── confusion_matrices/
│   ├── error_analysis.csv
│   └── metrics.csv
├── src/
│   ├── classical_models.py
│   ├── config.py
│   ├── dataset_utils.py
│   ├── evaluation.py
│   ├── final_analysis.py
│   ├── neural_models.py
│   ├── prediction.py
│   ├── preprocessing.py
│   ├── train_neural_models.py
│   └── word_embeddings.py
├── app.py                     Streamlit web application
├── run_project.py             command-line entry point
├── setup.ps1                  Windows environment setup
├── requirements.txt
└── README.md
```

Shared implementation is kept in `src/` so the notebooks, command-line tools,
and web application use the same processing and prediction logic.

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

Pretrained binary artifacts are not stored in the repository. Run the following
commands from the project root after installation:

```powershell
.\.venv\Scripts\python.exe run_project.py classical
.\.venv\Scripts\python.exe run_project.py embeddings
.\.venv\Scripts\python.exe run_project.py neural bilstm
.\.venv\Scripts\python.exe run_project.py neural transformer
.\.venv\Scripts\python.exe run_project.py finalize
```

These commands generate the four artifacts required by the ensemble:

```text
models/tfidf_logistic.pkl
models/word2vec_sentiment.pkl
models/bilstm_best.pt
models/transformer_best.pt
```

## Run the web application

After the model artifacts are available, start Streamlit:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

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
.\.venv\Scripts\python.exe run_project.py predict "service ta valo na"
```

Example output:

```text
Model: Validation-weighted NLP ensemble
Predicted sentiment: Negative
Probabilities:
  Positive: ...
  Negative: ...
  Neutral : ...
  Mixed   : ...
```

## Reproduce the complete experiment

Run the stages in order to recreate data preparation, every model comparison,
and the final evaluation:

```powershell
.\.venv\Scripts\python.exe run_project.py preprocess
.\.venv\Scripts\python.exe run_project.py classical
.\.venv\Scripts\python.exe run_project.py embeddings
.\.venv\Scripts\python.exe run_project.py neural rnn
.\.venv\Scripts\python.exe run_project.py neural bilstm
.\.venv\Scripts\python.exe run_project.py neural transformer
.\.venv\Scripts\python.exe run_project.py finalize
```

The fixed training, validation, and test files in `data/splits/` support
consistent comparison across model families.

## Notebooks

The notebooks present the project in experimental order:

1. Text preprocessing and corpus analysis
2. Classical NLP baselines
3. Word2Vec document representations
4. Vanilla RNN and Bidirectional LSTM
5. Transformer encoder
6. Final comparison, ensemble evaluation, and error analysis

Start JupyterLab with:

```powershell
.\.venv\Scripts\python.exe -m jupyter lab
```

## Outputs

- `results/metrics.csv` contains the consolidated experiment comparison.
- `results/confusion_matrices/` contains class-level evaluation figures.
- `results/error_analysis.csv` contains incorrectly classified test examples for
  qualitative analysis.
- `models/` contains locally generated inference artifacts.

## Scope and limitations

This system is designed for the language patterns represented in BnSentMix.
Performance may decrease for other domains, dialects, spelling conventions, or
sentiment definitions. Predictions should therefore be interpreted as model
estimates rather than definitive judgments about a writer's intent.
