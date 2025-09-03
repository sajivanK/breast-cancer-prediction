
## 📄 README.md

```markdown
# 🩺 Breast Cancer Prediction App

A **machine learning web app** built with **Streamlit** to predict whether a tumor is **Benign** or **Malignant** using the **Breast Cancer Wisconsin Dataset**.  
The model is trained using **Logistic Regression** with Scikit-Learn and deployed via **Streamlit Cloud**.

---

## 🚀 Features
- Upload a CSV file with patient features.
- Predicts whether each case is **Benign** or **Malignant**.
- Provides prediction probabilities.
- Displays a bar chart summary of results.

---

## 📂 Project Structure
```

breast-cancer-prediction/
│── app.py                   # Streamlit app
│── breast\_cancer\_model.pkl  # Trained Logistic Regression model
│── scaler.pkl               # StandardScaler for preprocessing
│── requirements.txt         # Project dependencies
│── sample.csv               # Example benign cases
│── malignant\_sample.csv     # Example malignant cases
│── README.md                # Project documentation

````

---

## 🛠 Installation (Run Locally)

1. Clone the repo:
   ```bash
   git clone https://github.com/sajivanK/breast-cancer-prediction.git
   cd breast-cancer-prediction
````

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:

   ```bash
   streamlit run app.py
   ```

4. Open browser at 👉 `http://localhost:8501`

---

## 🌐 Deployment on Streamlit Cloud

1. Push your repo to GitHub.
2. Go to [Streamlit Cloud](https://share.streamlit.io).
3. Create a new app → select your repo → branch (`main`) → `app.py`.
4. Deploy 🚀

---

## 📊 Input Format

Your CSV must include the **30 feature columns** used in training:

```
radius_mean,texture_mean,perimeter_mean,area_mean,smoothness_mean,
compactness_mean,concavity_mean,concave points_mean,symmetry_mean,fractal_dimension_mean,
radius_se,texture_se,perimeter_se,area_se,smoothness_se,
compactness_se,concavity_se,concave points_se,symmetry_se,fractal_dimension_se,
radius_worst,texture_worst,perimeter_worst,area_worst,smoothness_worst,
compactness_worst,concavity_worst,concave points_worst,symmetry_worst,fractal_dimension_worst
```

✅ Check [`sample.csv`](./sample.csv) and [`malignant_sample.csv`](./malignant_sample.csv) for examples.

---

## 🧪 Example

### Sample Input (`sample.csv`)

| radius\_mean | texture\_mean | perimeter\_mean | area\_mean | smoothness\_mean | ... | fractal\_dimension\_worst |
| ------------ | ------------- | --------------- | ---------- | ---------------- | --- | ------------------------- |
| 12.0         | 14.0          | 78.0            | 450.0      | 0.08             | ... | 0.06                      |

### Sample Output

* Prediction: **Benign**
* Malignant Probability: `0.000009`
* Benign Probability: `0.999991`

---

## ⚡ Tech Stack

* Python
* Scikit-Learn
* Pandas, NumPy
* Streamlit
* Joblib

---

## 🌐 Live Demo

👉 [Try the App Here](https://breast-cancer-prediction-4nx6dogifwiws6jvn8omtx.streamlit.app)

---
## 👨‍💻 Author

Developed by **\SajivanK** 🎯
Undergraduate | Data Science & Machine Learning Enthusiast


