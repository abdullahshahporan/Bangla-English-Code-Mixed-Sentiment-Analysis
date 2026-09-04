# Bangla-English Code-Mixed Sentiment Analysis

An NLP Lab project that processes Bangla-English code-mixed text and predicts
one of four sentiments:

- Positive
- Negative
- Neutral
- Mixed

## Final NLP system

The teacher provides only one sentence through the web interface. There is no
model selector. Four NLP approaches analyze the same sentence and their
probabilities are combined into one final answer:

```text
Teacher's text
      ↓
Code-mixed preprocessing and tokenization
      ↓
┌──────────────┬───────────┬─────────┬─────────────┐
│ TF-IDF + LR  │ Word2Vec  │ BiLSTM  │ Transformer │
└──────────────┴───────────┴─────────┴─────────────┘
      ↓ validation-Macro-F1-weighted probability average
One final sentiment + four probabilities
```

Each model's validation Macro F1 is used as its ensemble weight. The test set is
not used to choose the weights.

## Corpus source

The corpus is **BnSentMix: A Diverse Bengali-English Code-Mixed Dataset for
Sentiment Analysis**, created by Sadia Alam, Md Farhan Ishmam, Navid Hasin Alvee,
Md Shahnewaz Siddique, Md Azam Hossain, and Abu Raihan Mostofa Kamal.

- [Official BnSentMix repository](https://github.com/Nishita2000/BnSentMix)
- [Published ACL paper](https://aclanthology.org/2025.loreslm-1.4/)
- Local raw corpus: `data/raw/dataset.csv`

The dataset authors collected user-generated text from YouTube comments,
Facebook comments, and e-commerce reviews. The local file contains 20,015 rows;
after missing, empty, and duplicate records are removed, 19,822 samples remain.

| Numeric label | Sentiment |
|---:|---|
| 0 | Positive |
| 1 | Negative |
| 2 | Neutral |
| 3 | Mixed |

## Clean project structure

```text
CodeMixed-Sentiment/
│
├── data/
│   ├── raw/
│   │   └── dataset.csv
│   ├── processed/
│   │   └── processed_sentiment.csv
│   └── splits/
│       ├── train.csv
│       ├── validation.csv
│       └── test.csv
│
├── notebooks/
│   ├── 01_preprocessing.ipynb
│   ├── 02_classical_models.ipynb
│   ├── 03_word_embeddings.ipynb
│   ├── 04_rnn_bilstm.ipynb
│   ├── 05_transformer.ipynb
│   └── 06_final_comparison.ipynb
│
├── models/
│   ├── tfidf_logistic.pkl
│   ├── word2vec_sentiment.pkl
│   ├── bilstm_best.pt
│   └── transformer_best.pt
│
├── results/
│   ├── metrics.csv
│   ├── error_analysis.csv
│   └── confusion_matrices/
│
├── src/                       reusable code used by notebooks and interface
├── app.py                     teacher input interface
├── run_project.py             module runner
├── setup.ps1                  automatic Python 3.12 setup
├── requirements.txt
└── README.md
```

The `src/` directory is required. It prevents the notebooks and interface from
containing repeated copies of the same code.

## First-time setup

Gensim does not currently install correctly with Python 3.14 on Windows. Use the
provided setup script, which creates a Python 3.12 environment:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

## interface

Start the interface from the project root:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the address displayed by Streamlit, normally:

```text
http://localhost:8501
```

The teacher should:

1. Enter a Bangla-English sentence.
2. Click **Analyze sentiment**.
3. Inspect the original text, processed text, and tokens.
4. See one combined final sentiment and four probabilities.

Example input:

```text
camera bhalo but battery kharap
```

## Check the saved project

The project has already been trained. You can check it without retraining:

```powershell
.\.venv\Scripts\python.exe run_project.py predict "service ta valo na"
```

Expected output contains:

```text
Model: Validation-weighted NLP ensemble
Predicted sentiment: Negative
```

The final ensemble result is stored in `results/metrics.csv` as experiment
`M6.1`. Current test results are:

```text
Accuracy: 0.7192
Macro F1: 0.6970
```

## Rebuild every NLP module

Run these only when the project must be trained again:

```powershell
.\.venv\Scripts\python.exe run_project.py preprocess
.\.venv\Scripts\python.exe run_project.py classical
.\.venv\Scripts\python.exe run_project.py embeddings
.\.venv\Scripts\python.exe run_project.py neural rnn
.\.venv\Scripts\python.exe run_project.py neural bilstm
.\.venv\Scripts\python.exe run_project.py neural transformer
.\.venv\Scripts\python.exe run_project.py finalize
```


