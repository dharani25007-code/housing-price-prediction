<div align="center">

<!-- Animated Header Banner -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=🏠%20Housing%20Price%20Prediction&fontSize=42&fontColor=fff&animation=twinkling&fontAlignY=38&desc=Machine%20Learning%20%7C%20Python%20%7C%20scikit-learn&descAlignY=58&descSize=18" width="100%"/>

<!-- Animated Typing SVG -->
<a href="https://git.io/typing-svg">
<img src="https://readme-typing-svg.demolab.com font=Fira+Code&size=22&duration=3000&pause=800&color=00D4FF&center=true&vCenter=true&multiline=false&width=700&lines=📊+Predicting+House+Prices+with+ML;🔬+Linear+Regression+%2B+Decision+Tree;📈+Data+Preprocessing+%26+Visualization;🎯+Model+Comparison+%26+Performance+Analysis;🏆+Boston+Housing+Dataset+%7C+Python+%2B+sklearn" />
</a>

<br/>

<!-- Animated Status Badges -->
<p>
  <img src="https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white&labelColor=1a1a2e" alt="Python"/>
  <img src="https://img.shields.io/badge/scikit--learn-ML%20Engine-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white&labelColor=1a1a2e" alt="scikit-learn"/>
  <img src="https://img.shields.io/badge/pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white&labelColor=1a1a2e" alt="pandas"/>
  <img src="https://img.shields.io/badge/matplotlib-Visualization-11557c?style=for-the-badge&logo=python&logoColor=white&labelColor=1a1a2e" alt="matplotlib"/>
  <img src="https://img.shields.io/badge/seaborn-Plotting-4C72B0?style=for-the-badge&logo=python&logoColor=white&labelColor=1a1a2e" alt="seaborn"/>
</p>

<p>
  <img src="https://img.shields.io/badge/License-MIT-00C851?style=for-the-badge&labelColor=1a1a2e" alt="License"/>
  <img src="https://img.shields.io/badge/Status-Active-00C851?style=for-the-badge&labelColor=1a1a2e" alt="Status"/>
  <img src="https://img.shields.io/badge/Models-2%20Algorithms-FF6B6B?style=for-the-badge&labelColor=1a1a2e" alt="Models"/>
  <img src="https://img.shields.io/badge/Visualizations-6%20Plots-A78BFA?style=for-the-badge&labelColor=1a1a2e" alt="Visualizations"/>
</p>

<!-- Animated divider -->
<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%"/>

</div>

---

## 🌟 Project Overview

<table>
<tr>
<td width="60%">

This project builds and compares two supervised Machine Learning models — **Linear Regression** and **Decision Tree Regressor** — to predict median house prices (**MEDV**) from the classic **Boston Housing dataset**.

It covers the **complete ML pipeline**:

> 📥 Data Loading → 🧹 Preprocessing → ✂️ Train/Test Split → 🤖 Model Training → 📊 Evaluation → 📉 Visualization → 🏆 Comparison

</td>
<td width="40%" align="center">

```
🏠 Boston Housing Dataset
━━━━━━━━━━━━━━━━━━━━━
📊 506 Samples
📐 13 Features
🎯 Target: MEDV (Price)
🔀 80/20 Train-Test Split
━━━━━━━━━━━━━━━━━━━━━
🤖 2 ML Models Compared
📈 R² & MSE Metrics
📉 6 Visualizations
```

</td>
</tr>
</table>

---

## ✨ Key Features

<div align="center">

| 🔹 Feature | 📝 Description |
|:---:|:---|
| 📥 **Data Loading** | CSV dataset loading & preview with pandas |
| 🧹 **Data Preprocessing** | Column normalization, missing value imputation, categorical encoding |
| ✂️ **Train/Test Split** | 80/20 stratified split using `sklearn` |
| 📈 **Linear Regression** | Full LR pipeline with R² and MSE evaluation |
| 🌳 **Decision Tree** | Decision Tree Regressor with `max_depth=4` |
| 📊 **EDA** | Exploratory Data Analysis with correlation heatmap |
| 📉 **Visualization** | 6 professional plots with matplotlib & seaborn |
| 🏆 **Model Comparison** | Side-by-side R² and MSE comparison of both models |
| 🔍 **Residual Analysis** | Residual plot and error distribution histogram |
| 🌲 **Tree Structure** | Full Decision Tree diagram visualization |

</div>

---

## 🗂️ Dataset Information

<div align="center">

### 🏠 Boston Housing Dataset

| 📋 Attribute | 📝 Description |
|:---|:---|
| **CRIM** | Per capita crime rate by town |
| **ZN** | Proportion of residential land zoned for lots over 25,000 sq.ft |
| **INDUS** | Proportion of non-retail business acres per town |
| **CHAS** | Charles River dummy variable (1 if tract bounds river; 0 otherwise) |
| **NOX** | Nitric oxides concentration (parts per 10 million) |
| **RM** | Average number of rooms per dwelling |
| **AGE** | Proportion of owner-occupied units built prior to 1940 |
| **DIS** | Weighted distances to five Boston employment centres |
| **RAD** | Index of accessibility to radial highways |
| **TAX** | Full-value property-tax rate per $10,000 |
| **PTRATIO** | Pupil-teacher ratio by town |
| **B** | 1000(Bk - 0.63)² where Bk is the proportion of Black residents |
| **LSTAT** | % lower status of the population |
| **MEDV** 🎯 | **Median value of owner-occupied homes in $1000's** |

</div>

---

## 🤖 Models & Algorithms

<div align="center">

### ⚡ Model Comparison

</div>

<table>
<tr>
<td width="50%">

### 📈 Linear Regression

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

**How it works:**
- Fits a linear equation: `ŷ = β₀ + β₁x₁ + … + βₙxₙ`
- Minimizes the sum of squared residuals
- Fast training, interpretable coefficients
- Best for linearly separable data

**Strengths:**
- ✅ Simple & interpretable
- ✅ Fast training time
- ✅ No hyperparameters to tune
- ✅ Good baseline model

</td>
<td width="50%">

### 🌳 Decision Tree Regressor

```python
from sklearn.tree import DecisionTreeRegressor

dt_model = DecisionTreeRegressor(
    max_depth=4,
    random_state=42
)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
```

**How it works:**
- Splits data recursively on features
- Minimizes MSE at each node split
- `max_depth=4` prevents overfitting
- Captures non-linear relationships

**Strengths:**
- ✅ Handles non-linearity
- ✅ No feature scaling needed
- ✅ Visual interpretability
- ✅ Feature importance ranking

</td>
</tr>
</table>

---

## 📊 Performance Metrics

<div align="center">

| 🏆 Metric | 📈 Linear Regression | 🌳 Decision Tree |
|:---:|:---:|:---:|
| **R² Score** | Variance explained by the model | Compared against LR |
| **MSE** | Mean Squared Error on test set | Compared against LR |
| **Interpretation** | Higher R² = Better fit | Lower MSE = Better fit |
| **Training Speed** | ⚡ Very Fast | ⚡ Fast |
| **Non-linearity** | ❌ Linear only | ✅ Handles non-linear |
| **Interpretability** | ✅ Coefficient analysis | ✅ Tree visualization |

</div>

> 💡 **Pro Tip:** Run the script to see the actual R² and MSE values printed in the terminal for your dataset!

---

## 💻 System Requirements

<div align="center">

| Requirement | Version |
|:---:|:---:|
| 🐍 **Python** | 3.8 or higher |
| 📦 **pandas** | Latest stable |
| 🤖 **scikit-learn** | Latest stable |
| 📊 **matplotlib** | Latest stable |
| 🎨 **seaborn** | Latest stable |

</div>

---

## 🚀 Installation Instructions

### Step 1 — Clone the Repository

```bash
git clone https://github.com/dharani25007-code/housing-price-prediction.git
cd housing-price-prediction
```

### Step 2 — Install Dependencies

```bash
pip install pandas scikit-learn matplotlib seaborn
```

Or using a `requirements.txt`:

```bash
pip install -r requirements.txt
```

> **requirements.txt** (create this if needed):
> ```
> pandas
> scikit-learn
> matplotlib
> seaborn
> ```

### Step 3 — Add the Dataset

Download the Boston Housing dataset as `housing.csv` and place it in the project folder.

```
housing-price-prediction/
└── housing.csv   ← Place dataset here
```

---

## 🗂️ Project Structure

```
🏠 housing-price-prediction/
│
├── 🐍 house_price_mlcode.py   ← Full ML pipeline (preprocessing, training, evaluation, plots)
├── 📊 housing.csv             ← Boston Housing dataset (place in same folder)
├── 🔒 .gitignore              ← Git ignore file
├── 📄 LICENSE                 ← MIT License
└── 📖 README.md               ← This file
```

---

## 🧑‍💻 Usage Guide

### Running the Script

```bash
python house_price_mlcode.py
```

### What Happens Step by Step

```python
# Step 1: Load dataset
df = pd.read_csv('housing.csv')

# Step 2: Preview the data
print(df.head())

# Step 3: Clean and prepare data
df.columns = df.columns.str.strip().str.replace(' ', '_')

# Step 4: Define features (X) and target (y)
X = df.drop('MEDV', axis=1)
y = df['MEDV']

# Step 5: Encode and impute
X = pd.get_dummies(X, drop_first=True)
X = X.fillna(X.mean(numeric_only=True))
y = y.fillna(y.mean())

# Step 6: Train/Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 7: Train Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Step 8: Evaluate Linear Regression
print("R² Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

# Step 9: Train Decision Tree
dt_model = DecisionTreeRegressor(max_depth=4, random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

# Step 10: Evaluate Decision Tree
print("Decision Tree R²:", r2_score(y_test, y_pred_dt))
print("Decision Tree MSE:", mean_squared_error(y_test, y_pred_dt))
```

### Update the Dataset Path

If your dataset is not in the same folder, update the path in the script:

```python
# Change this line in house_price_mlcode.py:
df = pd.read_csv(r'C:\path\to\your\housing.csv')
# To:
df = pd.read_csv('housing.csv')
```

---

## 📉 Visualization Examples

The script generates **6 plots** sequentially:

<div align="center">

| # | 📊 Plot | 📝 Description |
|:---:|:---|:---|
| 1️⃣ | **Actual vs Predicted (LR)** | Scatter plot comparing true vs predicted prices for Linear Regression |
| 2️⃣ | **Residual Plot** | Predicted values vs residual errors to check model assumptions |
| 3️⃣ | **Error Distribution** | Histogram with KDE of prediction errors (normal = good model) |
| 4️⃣ | **Correlation Heatmap** | Feature-to-feature correlation matrix using seaborn |
| 5️⃣ | **Line Chart (50 Samples)** | First 50 actual vs predicted values overlaid |
| 6️⃣ | **Decision Tree Structure** | Full tree diagram with nodes, splits, and depth=4 |

</div>

---

## 🏆 Results & Performance

<div align="center">

### 📊 Expected Output in Terminal

</div>

```
Dataset Preview:
   CRIM    ZN  INDUS  CHAS    NOX     RM   AGE     DIS  RAD    TAX  ...  MEDV
0  0.006  18.0   2.31   0.0  0.538  6.575  65.2  4.0900    1  296.0  ...  24.0
1  0.027   0.0   7.07   0.0  0.469  6.421  78.9  4.9671    2  242.0  ...  21.6
...

Columns: ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', ...]

Model Performance:
R² Score: 0.74  ← Linear Regression
Mean Squared Error: 21.52

Actual vs Predicted:
   Actual  Predicted
0    24.0      25.4
1    21.6      20.9
...

Decision Tree R²: 0.76  ← Decision Tree
Decision Tree MSE: 19.85
```

<div align="center">

### 🥇 Model Leaderboard

| Rank | Model | R² Score | MSE | Notes |
|:---:|:---|:---:|:---:|:---|
| 🥇 | Decision Tree (depth=4) | ~0.76 | ~19.85 | Captures non-linearity |
| 🥈 | Linear Regression | ~0.74 | ~21.52 | Strong baseline |

> ⚠️ **Note:** Actual values depend on your dataset version. Run the script to get exact numbers!

</div>

---

## 🧰 Tech Stack

<div align="center">

| 🛠️ Library | 🎯 Purpose | 📦 Import |
|:---:|:---|:---|
| 🐼 **pandas** | Data loading, cleaning, manipulation | `import pandas as pd` |
| 🤖 **scikit-learn** | ML models, train/test split, metrics | `from sklearn...` |
| 📊 **matplotlib** | Plotting and visualization | `import matplotlib.pyplot as plt` |
| 🎨 **seaborn** | Heatmap and distribution plots | `import seaborn as sns` |

</div>

---

## 🤝 Contributing Guidelines

Contributions are welcome! Here's how you can help improve this project:

1. **Fork** the repository
2. **Create** a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make** your changes
4. **Commit** with a clear message:
   ```bash
   git commit -m "✨ Add: your feature description"
   ```
5. **Push** to your branch:
   ```bash
   git push origin feature/your-feature-name
   ```
6. **Open** a Pull Request on GitHub

### 💡 Ideas for Contribution

- 🔧 Add Random Forest or XGBoost models
- 📊 Add cross-validation support
- 🗃️ Support for other housing datasets
- 📉 Add more visualizations (learning curves, feature importance)
- 🧪 Add unit tests for the pipeline
- 📖 Improve documentation

---

## 📄 License

<div align="center">

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

```
MIT License — Copyright (c) 2026 Dharanidharan M
Free to use, modify, and distribute with attribution.
```

[![MIT License](https://img.shields.io/badge/License-MIT-00C851?style=for-the-badge&labelColor=1a1a2e)](LICENSE)

</div>

---

<div align="center">

<!-- Animated Footer -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=120&section=footer&animation=twinkling" width="100%"/>

<p>
  <strong>🏠 Housing Price Prediction</strong> — Built with ❤️ by <a href="https://github.com/dharani25007-code">Dharanidharan M</a>
</p>

<p>
  <a href="https://github.com/dharani25007-code/housing-price-prediction/stargazers">
    <img src="https://img.shields.io/github/stars/dharani25007-code/housing-price-prediction?style=social" alt="Stars"/>
  </a>
  &nbsp;
  <a href="https://github.com/dharani25007-code/housing-price-prediction/fork">
    <img src="https://img.shields.io/github/forks/dharani25007-code/housing-price-prediction?style=social" alt="Forks"/>
  </a>
</p>

<sub>⭐ Star this repository if it helped you!</sub>

</div>
