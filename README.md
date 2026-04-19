# 🏠 Housing Price Prediction

> A machine learning project that predicts house prices using Linear Regression and Decision Tree models, with full data preprocessing, visualization, and model evaluation.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-Data-150458?logo=pandas&logoColor=white)
![matplotlib](https://img.shields.io/badge/matplotlib-Visualization-blue)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Overview

This project builds and compares two supervised ML models — **Linear Regression** and **Decision Tree Regressor** — to predict median house prices (MEDV) from the classic Boston Housing dataset. It covers the full ML pipeline: data loading, cleaning, feature engineering, training, evaluation, and visualization.

---

## ✨ Features

- 📥 CSV dataset loading with pandas
- 🧹 Data cleaning — column normalization, missing value imputation, categorical encoding
- ✂️ Train/test split (80/20) with `sklearn`
- 📈 **Linear Regression** model with R² and MSE evaluation
- 🌳 **Decision Tree Regressor** (max_depth=4) for comparison
- 📊 5 visualizations:
  - Actual vs Predicted scatter plot
  - Residual plot
  - Error distribution histogram
  - Feature correlation heatmap
  - Line chart of first 50 predicted vs actual values
- 🌲 Decision tree structure plot

---

## 🗂️ Project Structure

```
housing-price-prediction/
│
├── house_price_mlcode.py   # Full ML pipeline — preprocessing, training, evaluation, plots
├── housing.csv             # Dataset (Boston Housing — place in same folder)
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas scikit-learn matplotlib seaborn
```

### Usage

1. Download the Boston Housing dataset as `housing.csv` and place it in the project folder.
2. Update the file path in the script:
```python
df = pd.read_csv('housing.csv')   # update path if needed
```
3. Run the script:
```bash
python house_price_mlcode.py
```

---

## 📊 Models & Evaluation

| Model | Metric | Description |
|---|---|---|
| Linear Regression | R² Score | Proportion of variance explained |
| Linear Regression | MSE | Mean Squared Error on test set |
| Decision Tree | R² Score | Compared against linear model |
| Decision Tree | MSE | Compared against linear model |

---

## 📉 Visualizations

The script generates 6 plots sequentially:

1. **Actual vs Predicted** — scatter plot for Linear Regression
2. **Residual Plot** — predicted values vs errors
3. **Error Distribution** — histogram with KDE of prediction errors
4. **Correlation Heatmap** — feature-to-feature correlation matrix
5. **Line Chart** — first 50 actual vs predicted samples
6. **Decision Tree Structure** — full tree diagram (max_depth=4)

---

## 🧰 Tech Stack

| Library | Purpose |
|---|---|
| `pandas` | Data loading, cleaning, manipulation |
| `scikit-learn` | ML models, train/test split, metrics |
| `matplotlib` | Plotting and visualization |
| `seaborn` | Heatmap and distribution plots |

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
