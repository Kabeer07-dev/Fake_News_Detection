# Fake News Detection

A machine learning project that classifies news articles as **Real** or **Fake** using classic NLP preprocessing and a Logistic Regression classifier trained on a Bag-of-Words representation of article text.

## Overview

The notebook (`news.ipynb`) walks through a complete text-classification pipeline:

1. Load and label the real/fake news datasets
2. Merge, shuffle, and explore the combined data
3. Clean and normalize the article text
4. Vectorize the text with a Bag-of-Words model
5. Train a Logistic Regression classifier
6. Evaluate with a held-out test set and 5-fold cross-validation

## Dataset

Located in `News_dataset/`:

| File | Rows | Description |
|------|------|-------------|
| `True.csv` | 21,417 | Genuine news articles |
| `Fake.csv` | 23,481 | Fake news articles |

Each file has the columns: `title`, `text`, `subject`, `date`.

During preprocessing, the two files are merged into a single dataframe with a binary `is_fake` label (`0` = real, `1` = fake), shuffled, and de-duplicated (209 duplicate rows removed), leaving **44,689** labeled articles.

> Note: `True.csv` and `Fake.csv` are not included in version control by default due to their size (~55 MB and ~63 MB). Make sure they are present under `News_dataset/` before running the notebook.

## Pipeline

**Text preprocessing** (applied to a combined `title + text` field called `content`):
- Lowercasing
- Removing non-ASCII characters/emojis
- Removing punctuation
- Removing URLs
- Removing HTML tags
- Removing stopwords (NLTK)
- Lemmatization (NLTK `WordNetLemmatizer`, verb part-of-speech)

**Modeling:**
- Train/test split: 80% / 20%, stratified on `is_fake`
- Feature extraction: `CountVectorizer` (Bag-of-Words)
- Classifier: `LogisticRegression` (`max_iter=1000`)
- Validation: 5-fold cross-validation using a `Pipeline` of `CountVectorizer` + `LogisticRegression`

## Results

| Metric | Score |
|--------|-------|
| Test Accuracy | 99.73% |
| Train Accuracy | 99.99% |
| 5-Fold CV Mean Accuracy | 99.66% |

The train/test accuracy gap suggests the model may be picking up on dataset-specific artifacts (e.g. source/style differences between the real and fake article collections) rather than purely generalizable signals of misinformation — worth keeping in mind before applying the model to news from other sources.

## Tech Stack

- Python
- pandas, numpy
- matplotlib, seaborn
- nltk (stopwords, WordNet lemmatizer)
- scikit-learn (CountVectorizer, LogisticRegression, train_test_split, cross_val_score)

## Project Structure

```
Fake_News_Detection/
├── News_dataset/
│   ├── True.csv
│   └── Fake.csv
├── news.ipynb
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook / JupyterLab

### Installation

```bash
git clone https://github.com/Kabeer07-dev/Fake_News_Detection.git
cd Fake_News_Detection

pip install numpy pandas matplotlib seaborn nltk scikit-learn jupyter
```

Download the required NLTK data (also done inside the notebook):

```python
import nltk
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
```

### Usage

1. Make sure `True.csv` and `Fake.csv` are in the `News_dataset/` folder.
2. Launch Jupyter and open the notebook:

```bash
jupyter notebook news.ipynb
```

3. Run the cells in order to reproduce preprocessing, training, and evaluation.

## Possible Improvements

- Replace Bag-of-Words with TF-IDF or word embeddings
- Try additional models (SVM, Random Forest, Gradient Boosting) for comparison
- Add regularization/hyperparameter tuning to close the train/test accuracy gap
- Evaluate on an external, out-of-distribution news dataset to test generalization
- Wrap the trained model in a simple script or web app for interactive predictions

## Author

[Kabeer07-dev](https://github.com/Kabeer07-dev)