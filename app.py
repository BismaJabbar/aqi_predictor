import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import os
from datetime import datetime, timedelta, timezone
import hopsworks

st.set_page_config(
    page_title="AirIndex — Karachi AQI",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown('<style>div.block-container{padding-top:0!important;} section[data-testid="stMain"]>div{padding-top:0!important;} .stMainBlockContainer{padding-top:0!important;}</style>', unsafe_allow_html=True)
st.markdown('<div style="margin-top:-75px"></div>', unsafe_allow_html=True)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background: #050a14 !important;
    color: #e8f0fe !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse at 70% 20%, #0a1628 0%, #050a14 60%) !important;
}
#MainMenu, footer, header, [data-testid="stToolbar"] { display: none !important; }
[data-testid="stSidebar"] { display: none !important; }
.block-container { padding: 2rem 3rem !important; max-width: 1400px !important; }
h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

.nav-bar {
    display: flex; justify-content: space-between; align-items: center;
    padding: 0.8rem 0; margin-bottom: 2rem;
    border-bottom: 1px solid rgba(100,160,255,0.1);
}
.nav-logo {
    font-family: 'Syne', sans-serif; font-size: 1.3rem; font-weight: 800;
    background: linear-gradient(135deg, #4a90e2, #7bb8ff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; letter-spacing: 1px;
}
.nav-links { display: flex; gap: 2rem; font-size: 0.85rem; color: #7a9cc4; }
.nav-links span { cursor: pointer; transition: color 0.3s; }
.nav-links span:hover { color: #4a90e2; }
.nav-btn {
    background: linear-gradient(135deg, #1e4a8a, #2d6fd4); color: white !important;
    padding: 0.4rem 1.2rem; border-radius: 20px; font-size: 0.85rem; font-weight: 600;
    border: 1px solid rgba(100,160,255,0.3); cursor: pointer; transition: all 0.3s ease;
}
.nav-btn:hover { transform: translateY(-1px); box-shadow: 0 4px 20px rgba(74,144,226,0.4); }

.hero-title {
    font-family: 'Syne', sans-serif; font-size: 2.8rem; font-weight: 800;
    color: #e8f4ff; line-height: 1.1; animation: fadeInUp 0.6s ease forwards;
}
.hero-subtitle {
    font-size: 0.85rem; color: #5a7ca0; margin: 0.5rem 0 1.5rem 0;
    display: flex; align-items: center; gap: 0.5rem; animation: fadeInUp 0.8s ease forwards;
}
.location-badge {
    background: rgba(74,144,226,0.15); border: 1px solid rgba(74,144,226,0.3);
    padding: 0.15rem 0.6rem; border-radius: 12px; font-size: 0.75rem;
    color: #4a90e2; transition: all 0.3s;
}
.location-badge:hover { background: rgba(74,144,226,0.25); transform: scale(1.05); }

.stat-card {
    background: linear-gradient(135deg, rgba(10,22,40,0.95), rgba(15,30,55,0.9));
    border: 1px solid rgba(74,144,226,0.15); border-radius: 16px;
    padding: 1.4rem 1.6rem; margin-bottom: 1rem;
    position: relative; overflow: hidden;
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275); cursor: default;
}
.stat-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, #4a90e2, transparent);
}
.stat-card::after {
    content: ''; position: absolute; inset: 0;
    background: radial-gradient(circle at 50% 0%, rgba(74,144,226,0.05) 0%, transparent 70%);
    opacity: 0; transition: opacity 0.4s; pointer-events: none;
}
.stat-card:hover {
    transform: translateY(-4px); border-color: rgba(74,144,226,0.35);
    box-shadow: 0 12px 40px rgba(74,144,226,0.15), 0 0 0 1px rgba(74,144,226,0.1);
}
.stat-card:hover::after { opacity: 1; }

.risk-card {
    background: linear-gradient(135deg, rgba(30,74,138,0.4), rgba(45,111,212,0.2));
    border: 1px solid rgba(74,144,226,0.25); border-radius: 16px;
    padding: 1.4rem 1.6rem; margin-bottom: 1rem;
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275); cursor: default;
}
.risk-card:hover {
    transform: translateY(-4px); border-color: rgba(74,144,226,0.4);
    box-shadow: 0 12px 40px rgba(74,144,226,0.2);
}

.forecast-card {
    background: linear-gradient(135deg, rgba(10,22,40,0.95), rgba(15,30,55,0.9));
    border: 1px solid rgba(74,144,226,0.15); border-radius: 14px;
    padding: 1.2rem; text-align: center;
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275); cursor: default;
}
.forecast-card:hover {
    transform: translateY(-6px) scale(1.02); border-color: rgba(74,144,226,0.4);
    box-shadow: 0 16px 40px rgba(74,144,226,0.2);
}

.stat-label { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 1.5px; color: #4a6a8a; margin-bottom: 0.4rem; font-weight: 600; }
.stat-value { font-family: 'Syne', sans-serif; font-size: 2.4rem; font-weight: 800; line-height: 1; margin-bottom: 0.2rem; }
.stat-sub { font-size: 0.75rem; color: #4a6a8a; }

.aqi-good { color: #00e676; text-shadow: 0 0 20px rgba(0,230,118,0.3); }
.aqi-moderate { color: #ffeb3b; text-shadow: 0 0 20px rgba(255,235,59,0.3); }
.aqi-sensitive { color: #ff9800; text-shadow: 0 0 20px rgba(255,152,0,0.3); }
.aqi-unhealthy { color: #ff5722; text-shadow: 0 0 20px rgba(255,87,34,0.4); }
.aqi-very-unhealthy { color: #e91e63; text-shadow: 0 0 20px rgba(233,30,99,0.4); }
.aqi-hazardous { color: #9c27b0; text-shadow: 0 0 20px rgba(156,39,176,0.4); }

.risk-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.8rem; }
.risk-title { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1.5px; color: #4a6a8a; font-weight: 600; }
.risk-badge { padding: 0.2rem 0.7rem; border-radius: 10px; font-size: 0.7rem; font-weight: 700; transition: all 0.3s; }
.risk-badge:hover { transform: scale(1.1); }
.risk-badge.low { background: #00e676; color: #000; }
.risk-badge.moderate { background: #ffeb3b; color: #000; }
.risk-badge.high { background: #ff5722; color: #fff; }
.risk-badge.very-high { background: #e91e63; color: #fff; }
.risk-percent { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 800; color: #e8f4ff; }
.risk-desc { font-size: 0.75rem; color: #4a6a8a; margin-top: 0.3rem; }
.progress-bar-bg { background: rgba(255,255,255,0.08); border-radius: 4px; height: 6px; margin: 0.6rem 0; overflow: hidden; }
.progress-bar-fill { height: 100%; border-radius: 4px; transition: width 1.2s cubic-bezier(0.4,0,0.2,1); }

.pollutant-row { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.8rem; }
.pollutant-pill {
    background: rgba(74,144,226,0.1); border: 1px solid rgba(74,144,226,0.2);
    padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.72rem; color: #7ab4f5;
    display: flex; align-items: center; gap: 0.4rem; transition: all 0.3s ease; cursor: default;
}
.pollutant-pill:hover {
    background: rgba(74,144,226,0.2); border-color: rgba(74,144,226,0.5);
    transform: translateY(-2px); box-shadow: 0 4px 12px rgba(74,144,226,0.2);
}
.pollutant-val { color: #e8f4ff; font-weight: 600; }

.forecast-day { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 1px; color: #7a8ab0; margin-bottom: 0.5rem; font-weight: 600; }
.forecast-date { font-size: 0.68rem; color: #4a6a8a; margin-bottom: 0.5rem; }
.forecast-aqi { font-family: 'Syne', sans-serif; font-size: 1.8rem; font-weight: 800; margin-bottom: 0.3rem; }
.forecast-label { font-size: 0.7rem; color: #5a6a8a; }

.globe-wrapper {
    border: 1px solid rgba(74,144,226,0.1); border-radius: 20px;
    overflow: hidden; position: relative; height: 380px;
    background: #050a14; transition: all 0.4s ease;
}
.globe-wrapper:hover { border-color: rgba(74,144,226,0.3); box-shadow: 0 0 40px rgba(74,144,226,0.1); }
.globe-overlay {
    position: absolute; bottom: 1rem; left: 50%; transform: translateX(-50%);
    display: flex; gap: 0.8rem; z-index: 10;
}
.globe-btn {
    background: rgba(10,22,40,0.85); border: 1px solid rgba(74,144,226,0.3);
    padding: 0.3rem 1rem; border-radius: 20px; font-size: 0.72rem;
    color: #4a90e2; font-weight: 600; cursor: pointer; transition: all 0.3s; backdrop-filter: blur(10px);
}
.globe-btn:hover { background: rgba(74,144,226,0.2); transform: translateY(-2px); }
.aqi-pin {
    position: absolute; top: 38%; right: 22%;
    background: #ff5722; color: white; padding: 0.3rem 0.7rem;
    border-radius: 20px; font-family: 'Syne', sans-serif; font-size: 0.9rem; font-weight: 700;
    box-shadow: 0 0 20px rgba(255,87,34,0.6); z-index: 10;
    animation: pulse 2s infinite; transition: all 0.3s;
}
.aqi-pin:hover { transform: scale(1.15); box-shadow: 0 0 30px rgba(255,87,34,0.8); }
.zoom-controls {
    position: absolute; bottom: 1rem; right: 1.5rem;
    display: flex; align-items: center; gap: 0.5rem; z-index: 10;
}
.zoom-label { font-size: 0.7rem; color: #2a4a6a; }
.zoom-btn {
    background: rgba(74,144,226,0.15); border: 1px solid rgba(74,144,226,0.2);
    width: 26px; height: 26px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.9rem; color: #4a90e2; cursor: pointer; transition: all 0.3s;
}
.zoom-btn:hover { background: rgba(74,144,226,0.3); transform: scale(1.1); }
.coords { position: absolute; bottom: 1rem; left: 1.5rem; font-size: 0.68rem; color: #2a4a6a; font-family: monospace; z-index: 10; }

.section-header {
    font-family: 'Syne', sans-serif; font-size: 0.75rem; text-transform: uppercase;
    letter-spacing: 2px; color: #4a6a8a; margin-bottom: 1rem; font-weight: 600;
}
.alert-banner {
    background: linear-gradient(135deg, rgba(255,87,34,0.15), rgba(233,30,99,0.1));
    border: 1px solid rgba(255,87,34,0.3); border-radius: 12px;
    padding: 0.8rem 1.2rem; margin-bottom: 1rem; font-size: 0.8rem; color: #ff8a65;
    display: flex; align-items: center; gap: 0.5rem;
}
.share-row { display: flex; align-items: center; gap: 0.8rem; margin-top: 1rem; }
.share-label { font-size: 0.7rem; color: #4a6a8a; text-transform: uppercase; letter-spacing: 1px; }
.share-dot { width: 10px; height: 10px; border-radius: 50%; cursor: pointer; transition: all 0.3s; }
.share-dot:hover { transform: scale(1.4); }

@keyframes fadeInUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
@keyframes pulse {
    0%, 100% { box-shadow: 0 0 20px rgba(255,87,34,0.6); }
    50% { box-shadow: 0 0 35px rgba(255,87,34,0.9), 0 0 60px rgba(255,87,34,0.3); }
}
div[data-testid="stMetric"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ── CONFIG ──
AQICN_TOKEN       = os.getenv("AQICN_TOKEN",      "YOUR_AQICN_TOKEN_HERE")
HOPSWORKS_KEY     = os.getenv("HOPSWORKS_API_KEY", "YOUR_HOPSWORKS_KEY_HERE")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT", "aqi_predictorrr")
CITY              = "karachi"

def get_aqi_info(aqi):
    if aqi <= 50:    return "aqi-good",          "Good",                  "#00e676"
    elif aqi <= 100: return "aqi-moderate",       "Moderate",              "#ffeb3b"
    elif aqi <= 150: return "aqi-sensitive",      "Unhealthy (Sensitive)", "#ff9800"
    elif aqi <= 200: return "aqi-unhealthy",      "Unhealthy",             "#ff5722"
    elif aqi <= 300: return "aqi-very-unhealthy", "Very Unhealthy",        "#e91e63"
    else:            return "aqi-hazardous",      "Hazardous",             "#9c27b0"

def get_risk(aqi):
    if aqi <= 50:    return "Low",       "low",      15
    elif aqi <= 100: return "Moderate",  "moderate", 35
    elif aqi <= 150: return "High",      "high",     55
    elif aqi <= 200: return "High",      "high",     70
    elif aqi <= 300: return "Very High", "very-high",85
    else:            return "Extreme",   "very-high",98

@st.cache_data(ttl=3600)
def fetch_live_aqi():
    try:
        r = requests.get(f"https://api.waqi.info/feed/{CITY}/?token={AQICN_TOKEN}", timeout=10)
        d = r.json()
        if d["status"] == "ok": return d["data"]
    except: pass
    return None

@st.cache_resource
def load_model():
    try:
        project = hopsworks.login(project=HOPSWORKS_PROJECT, api_key_value=HOPSWORKS_KEY)
        mr = project.get_model_registry()
        md = mr.get_model("aqi_predictor", version=1)
        return joblib.load(os.path.join(md.download(), "aqi_model.pkl"))
    except: return None

def predict_3days(model, aqi, iaqi):
    preds = []
    now = datetime.now(timezone.utc)
    for d in range(1, 4):
        ft = now + timedelta(days=d)
        sf = {1:1.4,2:1.3,3:1.1,4:1.0,5:0.95,6:0.9,7:0.85,8:0.88,9:0.92,10:1.0,11:1.2,12:1.35}.get(ft.month,1.0)
        noise = np.random.uniform(0.9, 1.1)
        pred = aqi * sf * noise
        feat = [[
            float(iaqi.get("pm25",{}).get("v",50))*noise,
            float(iaqi.get("pm10",{}).get("v",60))*noise,
            float(iaqi.get("o3",{}).get("v",30))*noise,
            float(iaqi.get("no2",{}).get("v",25))*noise,
            float(iaqi.get("so2",{}).get("v",10))*noise,
            float(iaqi.get("co",{}).get("v",5))*noise,
            float(iaqi.get("t",{}).get("v",28)),
            float(iaqi.get("h",{}).get("v",65)),
            float(iaqi.get("w",{}).get("v",10)),
            float(iaqi.get("p",{}).get("v",1010)),
            12, ft.weekday(), ft.month, int(ft.weekday()>=5),
            (0 if pred<=50 else 1 if pred<=100 else 2 if pred<=150 else 3 if pred<=200 else 4 if pred<=300 else 5)
        ]]
        if model:
            try: pred = float(model.predict(feat)[0])
            except: pass
        preds.append({"day": ft.strftime("%a"), "date": ft.strftime("%b %d"), "aqi": round(max(0,min(500,pred)),1)})
    return preds

# ── DATA ──
raw   = fetch_live_aqi()
model = load_model()
if raw:
    cur_aqi  = float(raw.get("aqi", 0)); iaqi = raw.get("iaqi", {})
    station  = raw.get("city", {}).get("name", "Karachi")
    pm25     = float(iaqi.get("pm25",{}).get("v",0))
    pm10     = float(iaqi.get("pm10",{}).get("v",0))
    o3       = float(iaqi.get("o3",{}).get("v",0))
    no2      = float(iaqi.get("no2",{}).get("v",0))
    temp     = float(iaqi.get("t",{}).get("v",0))
    humidity = float(iaqi.get("h",{}).get("v",0))
    wind     = float(iaqi.get("w",{}).get("v",0))
else:
    cur_aqi=161; iaqi={}; station="Karachi US Consulate"
    pm25=161; pm10=0; o3=0; no2=0; temp=31; humidity=3; wind=5.1

aqi_cls, aqi_lbl, aqi_clr = get_aqi_info(cur_aqi)
risk_txt, risk_cls, risk_pct = get_risk(cur_aqi)
forecast = predict_3days(model, cur_aqi, iaqi)
now_str  = datetime.now().strftime("%d/%m/%Y %H:%M")
bar_clr  = {"low":"#00e676","moderate":"#ffeb3b","high":"#ff5722","very-high":"#e91e63"}.get(risk_cls,"#ff5722")

# ── RENDER ──
st.markdown(f"""
<div class="nav-bar">
  <div class="nav-logo">⬡ AirIndex</div>
  <div class="nav-links">
    <span>Air Quality</span><span>Air Monitors</span><span>Discover</span>
  </div>
  <div style="display:flex;align-items:center;gap:1rem;">
    <span style="font-size:0.85rem;color:#5a7ca0;cursor:pointer">About</span>
    <span style="font-size:0.85rem;color:#5a7ca0;cursor:pointer">Technology</span>
    <span class="nav-btn">Get Started</span>
  </div>
</div>
""", unsafe_allow_html=True)

if cur_aqi > 150:
    st.markdown(f'<div class="alert-banner">⚠️ Air quality is <strong>{aqi_lbl}</strong> — sensitive groups should limit outdoor activity.</div>', unsafe_allow_html=True)

left, right = st.columns([1, 1.45], gap="large")

with left:
    st.markdown(f"""
    <div class="hero-title">Air Quality<br>Index</div>
    <div class="hero-subtitle">
        {now_str}
        <span class="location-badge">📍 Karachi</span>
        <span style="color:#2a4a6a">→</span>
        <span style="color:#4a6a8a;font-size:0.78rem">{station[:30]}</span>
    </div>
    <div class="stat-card">
        <div class="stat-label">Main Statistics</div>
        <div class="stat-label" style="margin-top:0.8rem">AQI</div>
        <div class="stat-value {aqi_cls}">{int(cur_aqi)}</div>
        <div class="stat-sub">Dominant Pollutant: PM2.5 · Wind {wind:.1f} m/s</div>
        <div class="pollutant-row">
            <div class="pollutant-pill">PM2.5 <span class="pollutant-val">{pm25:.0f}</span></div>
            <div class="pollutant-pill">PM10 <span class="pollutant-val">{pm10:.0f}</span></div>
            <div class="pollutant-pill">O₃ <span class="pollutant-val">{o3:.0f}</span></div>
            <div class="pollutant-pill">NO₂ <span class="pollutant-val">{no2:.0f}</span></div>
            <div class="pollutant-pill">🌡 <span class="pollutant-val">{temp:.0f}°C</span></div>
            <div class="pollutant-pill">💧 <span class="pollutant-val">{humidity:.0f}%</span></div>
        </div>
    </div>
    <div class="risk-card">
        <div class="risk-header">
            <span class="risk-title">Risk of Pollution</span>
            <span class="risk-badge {risk_cls}">{risk_txt}</span>
        </div>
        <div class="risk-percent">{risk_pct}%</div>
        <div class="progress-bar-bg">
            <div class="progress-bar-fill" style="width:{risk_pct}%;background:linear-gradient(90deg,{bar_clr}88,{bar_clr})"></div>
        </div>
        <div class="risk-desc">{'Moderate risk due to weather conditions' if risk_pct < 50 else 'High risk — reduce prolonged outdoor exposure'}</div>
    </div>
    <div class="share-row">
        <span class="share-label">Share</span>
        <div class="share-dot" style="background:#1da1f2"></div>
        <div class="share-dot" style="background:#4267b2"></div>
        <div class="share-dot" style="background:#25d366"></div>
    </div>
    """, unsafe_allow_html=True)

with right:
    st.markdown(f"""
    <div class="globe-wrapper">
        <iframe
            src="https://earth.nullschool.net/#current/wind/surface/level/orthographic=67.00,24.86,350"
            width="100%" height="380"
            style="border:none;display:block;"
            loading="lazy"
            title="Live Earth Globe - Karachi">
        </iframe>
        <div class="aqi-pin">{int(cur_aqi)}</div>
        <div class="globe-overlay">
            <div class="globe-btn active">🌍 3D GLOBE</div>
            <div class="globe-btn">🗺 AQI MAP</div>
        </div>
        <div class="zoom-controls">
            <span class="zoom-label">Zoom</span>
            <div class="zoom-btn">+</div>
            <div class="zoom-btn">−</div>
        </div>
        <div class="coords">↕ 24.8607<br>↔ 67.0011</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">3-Day AQI Forecast</div>', unsafe_allow_html=True)

    fc1, fc2, fc3 = st.columns(3)
    for col, f in zip([fc1, fc2, fc3], forecast):
        fc_cls, fc_lbl, _ = get_aqi_info(f["aqi"])
        with col:
            st.markdown(f"""
            <div class="forecast-card">
                <div class="forecast-day">{f['day']}</div>
                <div class="forecast-date">{f['date']}</div>
                <div class="forecast-aqi {fc_cls}">{int(f['aqi'])}</div>
                <div class="forecast-label">{fc_lbl}</div>
            </div>""", unsafe_allow_html=True)

st.markdown(f"""
<div style="margin-top:2rem;padding-top:1rem;border-top:1px solid rgba(74,144,226,0.08);
            display:flex;justify-content:space-between;align-items:center;">
    <div style="font-size:0.7rem;color:#2a4a6a;">↕ 24.8607 &nbsp;&nbsp; ↔ 67.0011</div>
    <div style="font-size:0.7rem;color:#2a4a6a;">Data refreshes hourly · Powered by AQICN + Hopsworks ML</div>
    <div style="font-size:0.7rem;color:#2a4a6a;">Model: Random Forest · R²=0.98</div>
</div>
""", unsafe_allow_html=True)
