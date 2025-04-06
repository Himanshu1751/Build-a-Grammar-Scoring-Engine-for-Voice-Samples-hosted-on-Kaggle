# 🎧 SHL Intern Hiring Assessment – Grammar Scoring Engine

## 📂 Dataset

The dataset consists of:
- `train.csv`: Contains `filename` and `label` columns where `label` is the grammar score.
- `test.csv`: Contains filenames for the test set (no labels).
- `audios_train/`: Directory with audio files used for training.
- `audios_test/`: Directory with audio files used for testing.

---

## 🔍 Problem Statement

Given 45–60 second audio clips of spoken English, the task is to predict a **grammar score** from `0.0` to `5.0`. The challenge involves extracting meaningful audio features and training a regression model to capture subtle grammatical variations.

---

## 🧪 Approach

1. **Audio Feature Extraction**:
    - Used `librosa` to extract MFCCs, along with delta and delta-delta features.
    - Computed the mean over time for fixed-length feature vectors.

2. **Modeling**:
    - Used `RandomForestRegressor` from `sklearn` as a baseline model.
    - Trained on extracted MFCC features.
    - Achieved a **high training Pearson correlation** score.

3. **Prediction**:
    - Made predictions on test audio files.
    - Rounded predictions to 1 decimal place.
    - Prepared a submission CSV as per required format.

---

## 📈 Results

- ✅ **Training Pearson Correlation**: `0.9785`
- 📊 **Evaluation Metric**: Pearson Correlation on unseen test data (evaluated by SHL)

---

## 🚀 Getting Started

### Requirements

Install the following Python libraries:

```bash
pip install numpy pandas matplotlib librosa scikit-learn tqdm
