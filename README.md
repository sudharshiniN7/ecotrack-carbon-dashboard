# 🌱 EcoTrack — Student Carbon Footprint Awareness Dashboard

An AI-powered, student-friendly web dashboard to visualize and understand carbon footprint data across a campus cohort.

---

## 📁 Project Structure

```
carbon_dashboard/
├── app.py                   # Flask backend + ML model (Linear Regression)
├── requirements.txt         # Python dependencies
├── README.md
├── dataset/
│   └── sample_data.csv      # 40-row sample student dataset
├── templates/
│   └── index.html           # Main dashboard UI (Chart.js powered)
└── static/
    └── style.css            # Dark eco-themed stylesheet
```

---

## ✨ Features

| Feature | Description |
|---|---|
| 📊 Dashboard | 7 interactive charts (doughnut, pie, bar, scatter, polar area) |
| 🧮 Calculator | ML-backed carbon footprint estimator with emission breakdown |
| 🏆 Leaderboard | Top 10 eco-champion students |
| 📂 Upload CSV | Replace dataset with your own campus data |
| 🤖 ML Model | Scikit-learn LinearRegression trained on lifestyle features |

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the server
```bash
python app.py
```

### 3. Open in browser
```
http://localhost:5000
```

---

## 📊 Dataset Columns

Your CSV must contain these columns (case-sensitive):

| Column | Example Values |
|---|---|
| `student_id` | S001, S002 |
| `name` | Aarav Sharma |
| `age` | 19 |
| `gender` | Male / Female |
| `role` | Undergraduate / Postgraduate / PhD |
| `transport_mode` | Bus / Bicycle / Walk / Public Transit / Motorcycle / Car |
| `diet_type` | Omnivore / Vegetarian / Vegan |
| `awareness_level` | High / Medium / Low |
| `weekly_travel_km` | 45 |
| `meat_meals_per_week` | 7 |
| `electricity_kwh_month` | 80 |
| `device_hours_per_day` | 6 |
| `carbon_footprint_kg` | 312 |

---

## 🤖 ML Model Details

- **Algorithm**: Linear Regression (scikit-learn)
- **Features used**: `weekly_travel_km`, `meat_meals_per_week`, `electricity_kwh_month`, `device_hours_per_day`, + encoded `transport_mode`, `diet_type`, `awareness_level`
- **Target**: `carbon_footprint_kg` (annual CO₂ in kg)
- Model auto-retrains when a new CSV is uploaded

---

## 🌍 Carbon Footprint Rating Scale

| Score | Rating | Meaning |
|---|---|---|
| < 150 kg | Excellent 🟢 | Green champion |
| 150–280 kg | Good 🟡 | Doing well |
| 280–400 kg | Moderate 🟠 | Room to improve |
| 400–520 kg | High 🔴 | Action needed |
| > 520 kg | Very High 🚨 | Urgent changes needed |

---

## 🛠️ Tech Stack

- **Frontend**: HTML5, CSS3, Vanilla JS, Chart.js 4
- **Backend**: Python 3.8+, Flask
- **ML**: scikit-learn, pandas, numpy

---

## 📸 Pages

1. **Dashboard** — KPI cards + 7 live charts
2. **Calculator** — Lifestyle inputs → CO₂ score + breakdown + tip
3. **Leaderboard** — Top eco-friendly students ranked

---

*Built for student sustainability awareness. Every action counts. 🌿*
