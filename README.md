# 🎓 Admission Prediction: Graduate Admission Chances using Multiple Linear Regression

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-orange)
![Model](https://img.shields.io/badge/Model-OLS%20Regression-green)
![Adj R²](https://img.shields.io/badge/Adj.%20R%C2%B2-0.791-brightgreen)

A statistical analysis and predictive model that estimates a candidate's **probability of graduate admission** from academic and qualitative profile factors, and identifies which factors matter most.

---

## 📌 Table of Contents
1. [Problem Statement](#-problem-statement)
2. [Dataset](#-dataset)
3. [Methodology](#-methodology)
4. [Key Results](#-key-results)
5. [Model Diagnostics](#-model-diagnostics)
6. [Limitations & Future Work](#-limitations--future-work)
7. [Repository Structure](#-repository-structure)
8. [How to Run](#-how-to-run)
9. [Tech Stack](#-tech-stack)
10. [Authors](#-authors)

---

## 🎯 Problem Statement

Institutions must evaluate applicants using a mix of quantitative metrics (GRE, CGPA) and qualitative indicators (SOP, recommendation letters). This project tackles that **selection uncertainty problem** by:

- Quantifying the impact of academic variables versus institutional and qualitative factors on admission chances.
- Building a Multiple Linear Regression model that turns several inputs into a single, interpretable probability score.
- Identifying the key **"success drivers"** behind an admission profile.

The same approach applies to other multivariate scoring problems such as recruitment analytics and credit-risk scoring.

## 📊 Dataset

- **Source:** Graduate Admissions dataset (`Admission_Predict.csv`), widely available on Kaggle.
- **Size:** 400 applicants, 7 predictors, 1 target (after dropping `Serial No.`).
- **Data quality:** no missing values (0.00%) and no duplicate rows.

| Feature | Description |
|---|---|
| `GRE Score` | Quantitative and verbal aptitude score |
| `TOEFL Score` | English-language proficiency score |
| `University Rating` | Rating of the target university (1 to 5) |
| `SOP` | Strength of Statement of Purpose (1 to 5) |
| `LOR` | Strength of Letter of Recommendation (1 to 5) |
| `CGPA` | Undergraduate cumulative GPA (out of 10) |
| `Research` | Research experience (0 = No, 1 = Yes) |
| **`Chance of Admit`** | **Target:** probability of admission (0 to 1) |

## 🔬 Methodology

1. **Data cleaning:** removed the ID column, standardised column names, checked for duplicates and missing values.
2. **Outlier handling:** inspected box plots, then applied `RobustScaler` (median and IQR based), which is less sensitive to outliers than standard scaling.
3. **Exploratory data analysis:**
   - Applicant distribution and average admission chance by University Rating.
   - Academic benchmarking (GRE, TOEFL, CGPA) across rating tiers.
   - Correlation heatmap: GRE and TOEFL are strongly correlated, as are SOP and University Rating.
4. **Hypothesis testing:**
   - **Independent t-test:** applicants with research experience have a significantly higher mean chance of admission (p < 0.001).
   - **One-way ANOVA:** mean chance of admission differs significantly across University Rating groups (p < 0.001).
5. **Modelling:** 80/20 train-test split (`random_state=42`) and an OLS regression fitted with `statsmodels`.
6. **Feature selection (backward elimination):** dropped the variable with the highest p-value one at a time: first `SOP`, then `University Rating`, until every remaining predictor was significant at 5%.
7. **Diagnostics:** VIF for multicollinearity, residuals vs. predictions for constant variance, and a Q-Q plot for normality of residuals.

## 🏆 Key Results

**Final model (fitted on 320 training records):**

| Metric | Value |
|---|---|
| R² | 0.794 |
| Adjusted R² | 0.791 |
| F-statistic | 241.8 (p ≈ 2.3e-105) |

| Predictor | Coefficient | p-value | Significant |
|---|---|---|---|
| Intercept | 0.7384 | < 0.001 | ✅ |
| **CGPA** | **0.1067** | < 0.001 | ✅ |
| GRE Score | 0.0319 | 0.005 | ✅ |
| TOEFL Score | 0.0275 | 0.008 | ✅ |
| Research | 0.0229 | 0.010 | ✅ |
| LOR | 0.0180 | 0.001 | ✅ |

**Fitted equation:**

```
Chance of Admit = 0.7384 + 0.0319·GRE + 0.0275·TOEFL + 0.0180·LOR + 0.1067·CGPA + 0.0229·Research
```

> ⚠️ **Reading the coefficients:** the predictors were scaled with `RobustScaler`, so each coefficient is the change in admission probability for a **one-IQR increase** in that feature (or for having research experience vs. not), not a change per raw GRE point or per 1.0 of CGPA. Raw inputs must be scaled the same way before using the equation for prediction.

**Main takeaways**
- **CGPA is the dominant driver** by a wide margin, roughly 3x the effect of GRE.
- GRE, TOEFL, LOR and Research each contribute a smaller but statistically significant effect.
- `SOP` and `University Rating` became statistically redundant once academic and research metrics were included, because they are correlated with those variables.
- The model explains about 79% of the variance in admission chances.

## 🩺 Model Diagnostics

| Check | Result |
|---|---|
| Multicollinearity (VIF) | All predictors below 5 (highest: CGPA 4.83, GRE 4.58) |
| Homoscedasticity | Residuals vs. predicted values show no strong pattern |
| Residual normality | Q-Q plot broadly follows the line (see limitations for a caveat) |
| Autocorrelation | Durbin-Watson ≈ 2.04, no sign of autocorrelation |

## ⚠️ Limitations & Future Work

Being upfront about scope:

- **Held-out evaluation:** the reported R² comes from the training data. A next step is scoring the untouched 20% test set (RMSE, MAE, test R²) and adding cross-validation.
- **Data leakage (minor):** the scaler was fitted on the full dataset before the split. Fitting it on the training set only, ideally inside a `Pipeline`, is the cleaner approach.
- **Residual normality:** the Q-Q plot is approximately normal, but the Omnibus test on the residuals is significant, suggesting mild departures (skew). Robust standard errors or a transformation could be explored.
- **Multicollinearity:** CGPA's VIF (4.83) is under the common threshold of 5 but close to it.
- **Small, observational data:** 400 rows. Findings describe association, not causation, and dropping `SOP` does not mean it is unimportant in real admissions.
- **Future work:** compare with Ridge/Lasso, Random Forest and Gradient Boosting; add a prediction script or a simple Streamlit app.

> This is an academic exercise and **should not be used for real admission decisions.**

## 📁 Repository Structure

```
Admission-Prediction/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── data/
│   └── Admission_Predict.csv
└── notebooks/
    └── graduate_admission_prediction.ipynb
```

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/saniaaa834/Admission-Prediction.git
cd Admission-Prediction

# 2. (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the notebook
jupyter notebook notebooks/graduate_admission_prediction.ipynb
```

The notebook reads the dataset from `../data/Admission_Predict.csv`. If you run it in Google Colab, upload the CSV and update the path in the first data-loading cell.

## 🛠️ Tech Stack

- **Language:** Python
- **Data handling:** Pandas, NumPy
- **Visualisation:** Matplotlib, Seaborn
- **Statistics and modelling:** SciPy, Statsmodels, Scikit-learn

## 👥 Authors

- Sania Sheikh (RBA46)

## 📄 License

Released under the [MIT License](LICENSE).
