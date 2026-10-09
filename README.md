<div align="center">

# 🧬 Parkinson's Disease Detection (XGBoost)

**Predict Parkinson's disease from voice measurements with XGBoost - includes a Streamlit prediction app.**

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-337AB7)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)

</div>

---

## ✨ Overview

- 📚 **Data:** the Oxford Parkinson's Disease Detection dataset (`parkinsons.data`, description in `parkinsons.names`) - biomedical voice measurements such as MDVP jitter, shimmer and fundamental frequency.
- 🧪 **Notebook (`app.ipynb`):** EDA, then **Logistic Regression** vs **XGBoost** classifiers. Accuracy on the held-out set is about **82-85 %**. Target: `0` = Parkinson's, `1` = healthy.
- 💾 The trained XGBoost model is saved as `xgb.pkl`.
- 🌐 **Streamlit app (`st.py`):** enter the voice measurements in the form and get a prediction.

> ⚠️ Educational project - not a medical diagnosis tool.

## 🚀 Getting Started

```bash
git clone https://github.com/Arashomranpour/Parkinson-xgboost.git
cd Parkinson-xgboost
pip install pandas numpy scikit-learn xgboost seaborn matplotlib streamlit joblib jupyter

jupyter notebook app.ipynb     # explore and (re)train
streamlit run st.py            # prediction app
```

## 📁 Project Structure

```
.
├── app.ipynb          # EDA + model training
├── st.py              # Streamlit prediction app
├── xgb.pkl            # Trained model
├── parkinsons.data    # Dataset
└── parkinsons.names   # Dataset description
```

## 🛠️ Tech Stack

`XGBoost` · `scikit-learn` · `pandas` · `Seaborn` · `Streamlit`
