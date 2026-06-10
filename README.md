# 🏠 Real Estate Price Prediction using Machine Learning

> **End-to-end regression pipeline to estimate property prices with Explainable AI**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-Model-brightgreen)](https://xgboost.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Overview

Real estate pricing is one of the most complex regression problems — driven by a mix of structural features, location dynamics, and market signals. This project builds a **production-style ML pipeline** to predict residential property prices using real-world Bangalore housing data.

The system covers the full data science lifecycle: **EDA → feature engineering → model benchmarking → hyperparameter tuning → deployment as a Streamlit web app** — with SHAP-based explainability to interpret predictions.

---

## 🔬 Key Features

- ✅ **8 Models Benchmarked** — Linear, Ridge, Lasso, ElasticNet, Decision Tree, Random Forest, Gradient Boosting, XGBoost
- ✅ **Advanced Feature Engineering** — Log transforms, frequency bucketing, ratio features, multicollinearity removal
- ✅ **SHAP Explainability** — Per-prediction feature attribution with waterfall + summary plots
- ✅ **Interactive Web App** — Streamlit app with dropdown, sliders, and price range output
- ✅ **Comprehensive EDA** — Choropleth maps, correlation heatmaps, distribution analysis

---

## 📂 Project Structure

```
real-estate-price-prediction/
├── data/
│   ├── raw/
│   │   └── Bengaluru_House_Data.csv    # Kaggle dataset (13,320 rows)
│   └── processed/
│       └── cleaned_features.csv
├── notebooks/
│   ├── 01_EDA.ipynb                    # Exploratory Data Analysis
│   ├── 02_Feature_Engineering.ipynb    # Preprocessing + feature creation
│   ├── 03_Model_Training.ipynb         # 8-model benchmark + CV
│   └── 04_SHAP_Explainability.ipynb    # SHAP analysis
├── src/
│   ├── preprocess.py                   # Data cleaning + feature engineering
│   ├── train.py                        # Model training + cross-validation
│   ├── evaluate.py                     # Metrics: R², MAE, RMSE, MAPE
│   ├── shap_explain.py                 # SHAP feature importance
│   └── predict.py                      # Inference function
├── models/
│   └── xgboost_best.pkl                # Saved best model (Pickle)
├── app/
│   └── streamlit_app.py                # Web app interface
├── requirements.txt
└── README.md
```

---

## 🧠 Technical Details

### Dataset
- **Source:** Kaggle — Bangalore House Price Dataset
- **Size:** 13,320 rows × 9 features
- **Raw Features:** `area_type`, `availability`, `location`, `size` (BHK), `society`, `total_sqft`, `bath`, `balcony`, `price`

### Feature Engineering Steps
| Step | Description |
|---|---|
| BHK Extraction | Parsed "2 BHK", "3 Bedroom" etc. → integer BHK value |
| Sqft Normalisation | Converted ranges like "1200-1400" → midpoint 1300 |
| Outlier Removal | Dropped records where price/sqft < 1% or > 99% percentile per location |
| Location Bucketing | 1000+ locations → top 240 (≥10 records) kept; rest → "Other" |
| Log Transform | Applied log1p to `price` and `total_sqft` to normalise skew |
| Ratio Feature | `price_per_sqft` as a learned regularisation anchor |
| Multicollinearity | Removed features with VIF > 10 |

### Model Performance Comparison
| Model | R² | MAE (₹L) | RMSE | MAPE |
|---|---|---|---|---|
| **XGBoost** | **0.88** | **4.2** | **6.1** | **8.3%** |
| Gradient Boosting | 0.86 | 4.6 | 6.5 | 9.1% |
| Random Forest | 0.84 | 5.1 | 7.0 | 10.2% |
| Decision Tree | 0.76 | 6.8 | 9.2 | 13.5% |
| Ridge Regression | 0.74 | 7.2 | 9.8 | 14.8% |
| Lasso | 0.73 | 7.4 | 10.0 | 15.1% |
| ElasticNet | 0.74 | 7.3 | 9.9 | 14.9% |
| Linear Regression | 0.71 | 7.8 | 10.6 | 16.3% |

*All models evaluated with 5-fold cross-validation*

### SHAP Feature Importance (Top 5)
1. `total_sqft` — 38% contribution
2. `location` (encoded) — 27%
3. `BHK` — 15%
4. `bath` — 12%
5. `price_per_sqft` (engineered) — 8%

---

## ⚙️ Installation & Setup

```bash
git clone https://github.com/sayali-babar19/real-estate-price-prediction.git
cd real-estate-price-prediction
pip install -r requirements.txt
```

### Train the Model
```bash
python src/train.py --model xgboost --cv 5
```

### Run Streamlit App
```bash
streamlit run app/streamlit_app.py
```

### Predict via Script
```python
from src.predict import predict_price

price = predict_price(
    location="Whitefield",
    bhk=3,
    total_sqft=1500,
    bath=2,
    balcony=1
)
print(f"Predicted Price: ₹{price:.1f} Lakhs")
```

---

## 📊 App Features

- 📍 **Location Dropdown** — 240+ Bangalore locations
- 🛏️ **BHK Selector** — 1 to 6 BHK
- 📐 **Area Slider** — 300 to 10,000 sqft
- 🚿 **Bath & Balcony Inputs**
- 💰 **Output** — Predicted price range with ±confidence interval
- 📊 **SHAP Waterfall** — Why did the model predict this price?

---

## 🔭 Future Work

- [ ] Add Pune, Mumbai, Hyderabad datasets for multi-city modelling
- [ ] Integrate live property listing scraper (99acres / MagicBricks)
- [ ] Add time-series price trend forecasting
- [ ] Deploy as REST API on AWS / GCP

---

## 👤 Author

**Sayali Babar**  
[https://www.linkedin.com/in/sayali-babar19/](https://www.linkedin.com/in/sayali-babar19/)  
Machine Learning Expert | AI & Neural Networks Practitioner | Data Analytics Practitioner

---

## 📄 License

This project is licensed under the **MIT License**. Feel free to use, modify, and distribute this project for educational and research purposes.

```
MIT License — Copyright (c) 2024 Sayali Girish Babar
```
