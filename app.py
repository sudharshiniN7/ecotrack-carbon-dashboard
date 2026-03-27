import os
import json
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

DATASET_PATH = os.path.join(os.path.dirname(__file__), 'dataset', 'sample_data.csv')

# ─────────────────────────────────────────────
# DATA LOADING & ML MODEL
# ─────────────────────────────────────────────

def load_data(filepath=None):
    path = filepath if filepath else DATASET_PATH
    df = pd.read_csv(path)
    return df

def train_model(df):
    """Train a simple linear regression model to predict carbon footprint."""
    le_transport = LabelEncoder()
    le_diet = LabelEncoder()
    le_awareness = LabelEncoder()

    df2 = df.copy()
    df2['transport_enc'] = le_transport.fit_transform(df2['transport_mode'])
    df2['diet_enc'] = le_diet.fit_transform(df2['diet_type'])
    df2['awareness_enc'] = le_awareness.fit_transform(df2['awareness_level'])

    features = ['weekly_travel_km', 'meat_meals_per_week',
                'electricity_kwh_month', 'device_hours_per_day',
                'transport_enc', 'diet_enc', 'awareness_enc']

    X = df2[features]
    y = df2['carbon_footprint_kg']

    model = LinearRegression()
    model.fit(X, y)

    return model, le_transport, le_diet, le_awareness

df_global = load_data()
model, le_transport, le_diet, le_awareness = train_model(df_global)

# ─────────────────────────────────────────────
# HELPER: chart-ready stats
# ─────────────────────────────────────────────

def get_dashboard_stats(df):
    transport_counts = df['transport_mode'].value_counts()
    transport_data = {'labels': transport_counts.index.tolist(), 'values': transport_counts.values.tolist()}

    diet_counts = df['diet_type'].value_counts()
    diet_data = {'labels': diet_counts.index.tolist(), 'values': diet_counts.values.tolist()}

    awareness_counts = df['awareness_level'].value_counts()
    awareness_data = {'labels': awareness_counts.index.tolist(), 'values': awareness_counts.values.tolist()}

    role_counts = df['role'].value_counts()
    role_data = {'labels': role_counts.index.tolist(), 'values': role_counts.values.tolist()}

    avg_carbon_transport = df.groupby('transport_mode')['carbon_footprint_kg'].mean().round(1)
    carbon_transport_data = {'labels': avg_carbon_transport.index.tolist(), 'values': avg_carbon_transport.values.tolist()}

    avg_carbon_diet = df.groupby('diet_type')['carbon_footprint_kg'].mean().round(1)
    carbon_diet_data = {'labels': avg_carbon_diet.index.tolist(), 'values': avg_carbon_diet.values.tolist()}

    kpis = {
        'total_students': int(len(df)),
        'avg_footprint': round(float(df['carbon_footprint_kg'].mean()), 1),
        'min_footprint': round(float(df['carbon_footprint_kg'].min()), 1),
        'max_footprint': round(float(df['carbon_footprint_kg'].max()), 1),
        'high_awareness_pct': round(float((df['awareness_level'] == 'High').sum() / len(df) * 100), 1),
        'eco_transport_pct': round(float(df['transport_mode'].isin(['Bicycle', 'Walk', 'Public Transit', 'Bus']).sum() / len(df) * 100), 1)
    }

    scatter_data = {
        'x': df['weekly_travel_km'].tolist(),
        'y': df['carbon_footprint_kg'].tolist(),
        'labels': df['name'].tolist(),
        'transport': df['transport_mode'].tolist()
    }

    return {
        'transport': transport_data, 'diet': diet_data, 'awareness': awareness_data,
        'role': role_data, 'carbon_by_transport': carbon_transport_data,
        'carbon_by_diet': carbon_diet_data, 'kpis': kpis, 'scatter': scatter_data
    }

# ─────────────────────────────────────────────
# AI SUGGESTIONS ENGINE
# ─────────────────────────────────────────────

def generate_suggestions(transport, diet, awareness, travel_km,
                          meat_meals, electricity, device_hours, predicted_kg):
    suggestions = []

    # ── TRANSPORT
    if transport == 'Car':
        saving = round(travel_km * (0.21 - 0.04) * 4 * 12, 0)
        suggestions.append({
            'text': 'Switch from your car to the bus or metro for daily commutes.',
            'detail': 'Cars emit ~5× more CO₂ per km than public buses. Even 2–3 bus days per week adds up significantly.',
            'saving_kg': int(saving * 0.6), 'icon': '🚌', 'category': 'Transport'
        })
        saving2 = round(travel_km * 0.21 * 4 * 12, 0)
        suggestions.append({
            'text': 'Consider carpooling with classmates for at least 3 days a week.',
            'detail': 'Sharing rides halves your per-person transport emissions immediately.',
            'saving_kg': int(saving2 * 0.5), 'icon': '🤝', 'category': 'Transport'
        })
    elif transport == 'Motorcycle':
        saving = round(travel_km * (0.11 - 0.04) * 4 * 12, 0)
        suggestions.append({
            'text': 'Replace motorcycle rides with public transit on shorter routes.',
            'detail': 'Motorbikes emit ~2.5× more than buses per passenger-km.',
            'saving_kg': int(saving * 0.5), 'icon': '🚇', 'category': 'Transport'
        })

    if transport not in ('Bicycle', 'Walk') and travel_km <= 60:
        suggestions.append({
            'text': 'For trips under 5 km, try cycling or walking instead.',
            'detail': 'Zero-emission travel for short trips is the easiest win available.',
            'saving_kg': int(travel_km * 0.15 * 4 * 12 * 0.3), 'icon': '🚲', 'category': 'Transport'
        })

    # ── DIET
    if diet == 'Omnivore':
        saving = round((meat_meals * 6.0) * 52 * 0.5, 0)
        suggestions.append({
            'text': 'Try going vegetarian 3–4 days per week (Meatless Mondays + more).',
            'detail': 'Beef alone generates 20× more greenhouse gas than lentils per gram of protein.',
            'saving_kg': int(saving), 'icon': '🥦', 'category': 'Diet'
        })
        if meat_meals >= 7:
            suggestions.append({
                'text': f'Reduce meat meals from {int(meat_meals)}/week to 3–4 by swapping in dal, tofu, or paneer.',
                'detail': 'Plant proteins require up to 10× less land and water than equivalent animal proteins.',
                'saving_kg': int(round((meat_meals - 4) * 6.0 * 52, 0)), 'icon': '🌱', 'category': 'Diet'
            })
    elif diet == 'Vegetarian' and meat_meals > 0:
        suggestions.append({
            'text': 'Replace remaining egg/dairy-heavy meals with plant-based alternatives a few times a week.',
            'detail': 'Dairy has a significant carbon footprint — oat or soy milk cuts it by ~70%.',
            'saving_kg': int(meat_meals * 2.5 * 52 * 0.4), 'icon': '🌿', 'category': 'Diet'
        })

    # ── ELECTRICITY
    if electricity > 80:
        saving = round((electricity - 60) * 0.7 * 12, 0)
        suggestions.append({
            'text': f'Reduce electricity use from {int(electricity)} to ~60 kWh/month by switching to LED bulbs and unplugging idle devices.',
            'detail': 'Phantom loads (devices on standby) account for 5–10% of household electricity.',
            'saving_kg': int(saving), 'icon': '💡', 'category': 'Energy'
        })
    if electricity > 60:
        suggestions.append({
            'text': "Use your college's solar-charged sockets or green-energy labs whenever possible.",
            'detail': 'Switching even 30% of consumption to renewable sources cuts your electricity CO₂ by ~30%.',
            'saving_kg': int(round(electricity * 0.7 * 12 * 0.3, 0)), 'icon': '☀️', 'category': 'Energy'
        })

    # ── DEVICES
    if device_hours >= 8:
        saving = round((device_hours - 5) * 30 * 0.05 * 12, 0)
        suggestions.append({
            'text': f'Cut screen time from {int(device_hours)} to ~5 hrs/day — enable "dark mode" and power-saving settings.',
            'detail': 'Laptops in power-saving mode use up to 60% less energy than in performance mode.',
            'saving_kg': int(saving), 'icon': '📵', 'category': 'Devices'
        })
    elif device_hours >= 5:
        suggestions.append({
            'text': 'Enable battery-saver / eco mode on all devices to cut idle power consumption.',
            'detail': 'Small efficiency gains across millions of students create meaningful collective impact.',
            'saving_kg': int(device_hours * 30 * 0.05 * 12 * 0.2), 'icon': '🔋', 'category': 'Devices'
        })

    # ── AWARENESS
    if awareness == 'Low':
        suggestions.append({
            'text': 'Join your campus sustainability club or attend one eco-workshop this semester.',
            'detail': 'Students who engage with sustainability communities reduce their footprint 15–25% on average within a year.',
            'saving_kg': int(predicted_kg * 0.15), 'icon': '🧠', 'category': 'Awareness'
        })
        suggestions.append({
            'text': 'Download a carbon-tracking app (e.g., Giki Zero or Capture) to log daily habits.',
            'detail': 'Tracking alone typically leads to a 10–12% voluntary reduction in emissions.',
            'saving_kg': int(predicted_kg * 0.10), 'icon': '📊', 'category': 'Awareness'
        })
    elif awareness == 'Medium':
        suggestions.append({
            'text': 'Share your carbon footprint score with friends and challenge them to do the same.',
            'detail': 'Social accountability is one of the strongest drivers of sustained behaviour change.',
            'saving_kg': int(predicted_kg * 0.08), 'icon': '📢', 'category': 'Awareness'
        })

    # ── HIGH FOOTPRINT NUDGE
    if predicted_kg > 400:
        suggestions.append({
            'text': 'Aim for a combined 20% reduction across transport + diet — the two biggest contributors.',
            'detail': "Transport and diet together typically account for 70–80% of a student's carbon footprint.",
            'saving_kg': int(predicted_kg * 0.20), 'icon': '🎯', 'category': 'Overall'
        })

    seen = set()
    unique = []
    for s in suggestions:
        if s['text'] not in seen:
            seen.add(s['text'])
            unique.append(s)

    unique.sort(key=lambda x: x['saving_kg'], reverse=True)
    return unique[:5]


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/dashboard', methods=['GET'])
def dashboard():
    stats = get_dashboard_stats(df_global)
    return jsonify(stats)

@app.route('/api/upload', methods=['POST'])
def upload():
    global df_global, model, le_transport, le_diet, le_awareness
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Only CSV files supported'}), 400
    try:
        df_global = pd.read_csv(file)
        model, le_transport, le_diet, le_awareness = train_model(df_global)
        stats = get_dashboard_stats(df_global)
        return jsonify({'success': True, 'data': stats})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/calculate', methods=['POST'])
def calculate():
    data = request.get_json()
    try:
        transport = data.get('transport_mode', 'Bus')
        diet = data.get('diet_type', 'Omnivore')
        awareness = data.get('awareness_level', 'Medium')
        travel_km = float(data.get('weekly_travel_km', 40))
        meat_meals = float(data.get('meat_meals_per_week', 5))
        electricity = float(data.get('electricity_kwh_month', 70))
        device_hours = float(data.get('device_hours_per_day', 6))

        def safe_encode(encoder, val):
            if val in encoder.classes_:
                return encoder.transform([val])[0]
            return 0

        t_enc = safe_encode(le_transport, transport)
        d_enc = safe_encode(le_diet, diet)
        a_enc = safe_encode(le_awareness, awareness)

        X_input = np.array([[travel_km, meat_meals, electricity, device_hours, t_enc, d_enc, a_enc]])
        predicted = float(model.predict(X_input)[0])
        predicted = max(0, round(predicted, 1))

        transport_co2_map = {
            'Car': travel_km * 0.21 * 4, 'Motorcycle': travel_km * 0.11 * 4,
            'Bus': travel_km * 0.04 * 4, 'Public Transit': travel_km * 0.035 * 4,
            'Bicycle': 0, 'Walk': 0
        }
        transport_co2 = transport_co2_map.get(transport, travel_km * 0.05 * 4)
        diet_co2 = meat_meals * 6.0
        electricity_co2 = electricity * 0.7
        device_co2 = device_hours * 30 * 0.05

        breakdown = {
            'Transport': round(transport_co2, 1), 'Diet': round(diet_co2, 1),
            'Electricity': round(electricity_co2, 1), 'Devices': round(device_co2, 1)
        }

        if predicted < 150:
            rating, rating_color = 'Excellent', '#22c55e'
            tip = "You're a green champion! Keep inspiring others around you."
        elif predicted < 280:
            rating, rating_color = 'Good', '#84cc16'
            tip = "You're doing well! Try switching to a bicycle or plant-based diet for more impact."
        elif predicted < 400:
            rating, rating_color = 'Moderate', '#f59e0b'
            tip = "Consider using public transport more and reducing meat consumption."
        elif predicted < 520:
            rating, rating_color = 'High', '#ef4444'
            tip = "Try carpooling, using buses, and cutting down on car trips and meat meals."
        else:
            rating, rating_color = 'Very High', '#dc2626'
            tip = "Urgent action needed! Switch to eco-transport and consider a vegetarian diet."

        suggestions = generate_suggestions(
            transport, diet, awareness,
            travel_km, meat_meals, electricity, device_hours, predicted
        )

        return jsonify({
            'predicted_kg': predicted, 'breakdown': breakdown,
            'rating': rating, 'rating_color': rating_color,
            'tip': tip, 'suggestions': suggestions
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/leaderboard', methods=['GET'])
def leaderboard():
    top = df_global.nsmallest(10, 'carbon_footprint_kg')[
        ['name', 'role', 'transport_mode', 'diet_type', 'carbon_footprint_kg']
    ].to_dict(orient='records')
    return jsonify(top)

if __name__ == '__main__':
    print("🌱 Carbon Footprint Dashboard running at http://localhost:5000")
    app.run(debug=True, port=5000)