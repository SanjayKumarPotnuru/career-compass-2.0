# 💼 Career Compass 2.0

> A smart web app that recommends the top 4 most suitable Computer Science job roles based on your **expected salary** and company **rating**, powered by machine learning.

---

## 🚀 Features

- 🔐 User registration & login
- 📈 ML-based prediction of top 4 job roles (e.g., SDE, Android, Backend)
- 🧠 Trained ensemble model using VotingClassifier (RandomForest, GradientBoosting, XGBoost)
- 📚 Job-wise detail pages (description, salary info, qualifications, scope, etc.)
- 🎯 Clean UI with dynamic routing
- 🌐 Deployed using **Render**

---

## 🖼️ Demo

🔗 [Live Render App Link](https://your-render-url.onrender.com) *(Replace after deployment)*

---

## 📂 Project Structure
Career-Compass-2.0/
├── app.py
├── models.py
├── requirements.txt
├── Procfile
├── /ml_model/
│ ├── job_predictor.py
│ ├── train_model.py
│ ├── final_ensemble_model_compressed.pkl
│ └── /data/
│ ├── tech_jobs.csv
│ └── job_info.json
├── /templates/
│ ├── base.html
│ ├── index.html
│ ├── dashboard.html
│ ├── login.html
│ ├── register.html
│ ├── results.html
│ ├── job_detail.html
│ └── about.html
├── /static/
│ ├── style.css
│ └── script.js



---

## ⚙️ Tech Stack

- **Frontend**: HTML5, CSS3, JS (Jinja2 templates)
- **Backend**: Flask + SQLAlchemy + SQLite
- **ML**: Scikit-learn, XGBoost, Imbalanced-learn
- **Deployment**: Render + Gunicorn

---

## 🧠 ML Model Details

- **Features used**: `Salary`, `Company Rating`
- **Target**: `Job Roles` (11 core CSE jobs)
- **Model**: Ensemble VotingClassifier (RF + GB + XGBoost)
- **Imbalance Handling**: SMOTE
- **Saved Model**: `final_ensemble_model_compressed.pkl` (joblib-compressed)

---

## ⚙️ How to Run Locally


# 1. Clone the repo
git clone https://github.com/yourusername/Career-Compass-2.0.git
cd Career-Compass-2.0

# 2. Create virtual env and activate
python -m venv env
env\Scripts\activate    # Windows
# source env/bin/activate  # Mac/Linux

# 3. Install requirements
pip install -r requirements.txt

# 4. Run the app
python app.py


