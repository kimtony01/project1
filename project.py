import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
import json
import os
import requests
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="해상 항로 리스크 조기경보 시스템",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------
# 스타일
# -------------------------------------------------------------
st.markdown("""
<style>
    .stApp { background-color: #0f172a; }
    h1 { color: #f8fafc !important; font-weight: 800 !important; letter-spacing: -0.5px; }
    h2, h3 { color: #e2e8f0 !important; font-weight: 700 !important; }
    .stCaption, p, span, label { color: #94a3b8 !important; }
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #1e293b, #172033);
        border: 1px solid #334155; border-radius: 14px;
        padding: 18px 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.25);
    }
    div[data-testid="stMetricLabel"] { color: #94a3b8 !important; font-size: 0.85rem !important; }
    div[data-testid="stMetricValue"] { color: #f8fafc !important; font-weight: 800 !important; }
    section[data-testid="stSidebar"] { background-color: #0b1220; border-right: 1px solid #1e293b; }
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
        color: #f8fafc !important; font-size: 1.0rem !important;
    }
    div[data-testid="stAlert"] { border-radius: 12px; border: none; }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #1e293b; border-radius: 14px; border: 1px solid #334155; padding: 8px 4px;
    }
    hr { border-color: #1e293b !important; }
    .stButton button {
        background-color: #3b82f6; color: white; border-radius: 10px; border: none; font-weight: 600;
    }
    .stButton button:hover { background-color: #2563eb; }
    div[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
    a { color: #60a5fa !important; text-decoration: none !important; }
    .risk-badge {
        display: inline-block; padding: 4px 14px; border-radius: 20px;
        font-size: 0.85rem; font-weight: 600; margin-top: 6px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 방향 화살표 헬퍼
# -------------------------------------------------------------
def calc_bearing(lat1, lon1, lat2, lon2):
    """두 좌표 사이의 방위각(도) 계산 - 화살표 회전 각도로 사용"""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    bearing = np.degrees(np.arctan2(x, y))
    return (bearing + 360) % 360

def add_direction_arrows(fig, lat_list, lon_list, color, size=11, n_arrows=3):
    """경로를 따라 균등한 지점에 진행 방향을 가리키는 삼각형 마커 추가"""
    n_points = len(lat_list)
    if n_points < 2:
        return
    positions = np.linspace(0, n_points - 2, n_arrows).astype(int)
    arrow_lat, arrow_lon, arrow_angle = [], [], []
    for i in positions:
        lat1, lon1 = lat_list[i], lon_list[i]
        lat2, lon2 = lat_list[i + 1], lon_list[i + 1]
        mid_lat = (lat1 + lat2) / 2
        mid_lon = (lon1 + lon2) / 2
        bearing = calc_bearing(lat1, lon1, lat2, lon2)
        arrow_lat.append(mid_lat)
        arrow_lon.append(mid_lon)
        arrow_angle.append(bearing)

    fig.add_trace(go.Scattergeo(
        lat=arrow_lat, lon=arrow_lon,
        mode="markers",
        marker=dict(
            symbol="triangle-up", size=size, color=color,
            angle=arrow_angle, angleref="up",
            line=dict(width=1, color="white")
        ),
        showlegend=False, hoverinfo="skip"
    ))

# -------------------------------------------------------------
# 데이터 정의
# -------------------------------------------------------------
ROUTE_DB = {
    "파나마 운하 (아시아-미주)": {
        "news_keyword": "Panama Canal shipping",
        "standard_risk_factors": {"news_cnt": 15, "weather_dist_km": 300, "threat_level": 3},
        "ship_location": {"lat": 9.38, "lon": -79.92, "name": "파나마 운하 대기 구역"},
        "threat_zone": {"lat": 9.08, "lon": -79.68, "name": "가툰 호수 가뭄/통항 제한 구역"},
        "standard_path": {
            "lat": [35.1, 33.0, 30.0, 25.0, 20.0, 15.0, 10.0, 8.9, 9.35, 15.0, 20.0, 25.0, 30.0, 35.0, 40.7],
            "lon": [129.0, 140.0, 155.0, 175.0, -170.0, -140.0, -100.0, -79.7, -79.9, -77.0, -80.0, -78.0, -75.0, -73.0, -74.0]
        },
        "alternatives": [
            {"route_name": "우회: 수에즈 운하 경유", "transit_time_days": 34, "cost_index_pct": 130, "safety_score": 80,
             "war_risk_insurance": "할증 가능",
             "eligibility": {"한국": "통행 가능", "미국": "조건부", "중국": "통행 가능", "영국": "조건부", "이스라엘": "통행 불가"},
             "status": "추천 우회로",
             "recommendation_reason": "대기 지연 없이 정시 도착 보장, 대형선 운항에 안정적.",
             "path_lat": [35.1, 25.0, 15.0, 5.0, 1.3, 5.0, 12.5, 20.0, 27.0, 31.0, 34.0, 36.0, 38.0, 40.7],
             "path_lon": [129.0, 120.0, 110.0, 105.0, 103.8, 90.0, 44.0, 38.0, 34.0, 32.3, 20.0, -5.6, -30.0, -74.0]}
        ]
    },
    "수에즈 / 홍해 (아시아-유럽)": {
        "news_keyword": "Red Sea shipping attack",
        "standard_risk_factors": {"news_cnt": 28, "weather_dist_km": 800, "threat_level": 4},
        "ship_location": {"lat": 12.8, "lon": 44.5, "name": "아덴만 진입부"},
        "threat_zone": {"lat": 14.5, "lon": 42.5, "name": "홍해 남부 분쟁 위험 구역"},
        "standard_path": {
            "lat": [35.1, 25.0, 15.0, 5.0, 1.3, 5.0, 12.5, 16.0, 20.0, 27.0, 31.0, 34.0, 36.0, 40.0, 45.0, 51.9],
            "lon": [129.0, 120.0, 110.0, 105.0, 103.8, 90.0, 44.0, 40.0, 38.0, 34.0, 32.3, 20.0, -5.6, -12.0, -8.0, 4.3]
        },
        "alternatives": [
            {"route_name": "우회: 아프리카 희망봉", "transit_time_days": 38, "cost_index_pct": 142, "safety_score": 95,
             "war_risk_insurance": "일반 요율",
             "eligibility": {"한국": "통행 가능", "미국": "통행 가능", "영국": "통행 가능", "중국": "통행 가능", "이스라엘": "통행 가능"},
             "status": "최우선 권장",
             "recommendation_reason": "13일 추가되지만 피격 위험 0%, 전쟁보험료 면제.",
             "path_lat": [35.1, 20.0, 5.0, 1.3, -10.0, -20.0, -30.0, -34.8, -30.0, -15.0, 0.0, 15.0, 25.0, 36.0, 40.0, 45.0, 51.9],
             "path_lon": [129.0, 115.0, 100.0, 103.8, 60.0, 45.0, 25.0, 20.0, 12.0, 8.0, -5.0, -12.0, -13.0, -9.5, -15.0, -8.0, 4.3]}
        ]
    },
    "호르무즈 해협 (중동-동아시아)": {
        "news_keyword": "Strait of Hormuz tension",
        "standard_risk_factors": {"news_cnt": 18, "weather_dist_km": 500, "threat_level": 4},
        "ship_location": {"lat": 26.5, "lon": 56.5, "name": "호르무즈 해협 진입부"},
        "threat_zone": {"lat": 26.8, "lon": 55.8, "name": "호르무즈 북부 군사 긴장 구역"},
        "standard_path": {
            "lat": [27.0, 24.0, 20.0, 15.0, 10.0, 5.0, 1.3, 5.0, 12.0, 18.0, 22.0, 27.0, 33.0, 35.1],
            "lon": [56.5, 60.0, 65.0, 68.0, 75.0, 85.0, 103.8, 105.0, 110.0, 113.0, 118.0, 122.0, 127.0, 129.0]
        },
        "alternatives": [
            {"route_name": "대체: 얀부항(홍해) 파이프라인 연계", "transit_time_days": 19, "cost_index_pct": 135, "safety_score": 85,
             "war_risk_insurance": "일반 요율",
             "eligibility": {"한국": "통행 가능", "미국": "통행 가능", "영국": "통행 가능", "중국": "통행 가능", "이스라엘": "조건부"},
             "status": "파이프라인 연계안",
             "recommendation_reason": "호르무즈 완전 우회, 봉쇄 시 유일한 현실적 대체 공급망.",
             "path_lat": [24.1, 18.0, 12.5, 8.0, 3.0, 1.3, 5.0, 12.0, 18.0, 22.0, 27.0, 33.0, 35.1],
             "path_lon": [38.0, 41.0, 44.0, 55.0, 70.0, 103.8, 105.0, 110.0, 113.0, 118.0, 122.0, 127.0, 129.0]}
        ]
    },
    "말라카 해협 (동남아-동아시아)": {
        "news_keyword": "Strait of Malacca shipping",
        "standard_risk_factors": {"news_cnt": 4, "weather_dist_km": 120, "threat_level": 2},
        "ship_location": {"lat": 2.5, "lon": 101.8, "name": "말라카 해협 중앙"},
        "threat_zone": {"lat": 4.0, "lon": 100.5, "name": "열대성 폭풍/해적 빈발 구역"},
        "standard_path": {
            "lat": [1.3, 3.0, 6.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.1],
            "lon": [103.8, 101.5, 103.0, 108.0, 112.0, 117.0, 121.0, 125.0, 129.0]
        },
        "alternatives": [
            {"route_name": "우회: 순다/롬복 해협", "transit_time_days": 12, "cost_index_pct": 125, "safety_score": 90,
             "war_risk_insurance": "일반 요율",
             "eligibility": {"한국": "통행 가능", "미국": "통행 가능", "중국": "통행 가능", "영국": "통행 가능", "이스라엘": "통행 가능"},
             "status": "기상 회피안",
             "recommendation_reason": "초강력 태풍 발생 시 동인도네시아 심해 수로로 안전 항해.",
             "path_lat": [-6.0, -8.7, -6.0, -3.0, 2.0, 5.0, 10.0, 18.0, 25.0, 35.1],
             "path_lon": [105.5, 115.8, 119.0, 122.0, 124.0, 126.0, 127.0, 128.0, 128.5, 129.0]}
        ]
    }
}

def calc_rule_score(news_cnt, weather_dist, threat_level):
    return min(100, int((news_cnt * 1.2) + max(0, (600 - weather_dist) * 0.08) + (threat_level * 10)))

# -------------------------------------------------------------
# 모델 & 뉴스
# -------------------------------------------------------------
@st.cache_resource
def get_trained_risk_model():
    np.random.seed(42)
    X = np.random.rand(400, 4)
    X[:, 0] *= 50; X[:, 1] *= 1000; X[:, 2] = (X[:, 2] - 0.5) * 200; X[:, 3] = np.random.randint(1, 6, 400)
    risk_score = (X[:, 0]*0.3) + ((1000-X[:, 1])*0.05) + (X[:, 2]*0.2) + (X[:, 3]*10)
    y = np.where(risk_score > 65, 2, np.where(risk_score > 35, 1, 0))
    model = RandomForestClassifier(n_estimators=30, random_state=42)
    model.fit(X, y)
    return model

model = get_trained_risk_model()

@st.cache_data(ttl=600)
def fetch_news(query, display=3):
    api_key = os.getenv("NEWSAPI_KEY")
    if not api_key:
        return None
    url = "https://newsapi.org/v2/everything"
    params = {"q": f'"{query}"', "language": "en", "sortBy": "publishedAt", "pageSize": display, "apiKey": api_key}
    try:
        res = requests.get(url, params=params, timeout=5)
        res.raise_for_status()
        articles = res.json().get("articles", [])
        return [{"title": a["title"], "link": a["url"], "desc": a.get("description", "") or ""} for a in articles]
    except Exception:
        return []

def predict_risk(news_cnt, weather_dist, news_growth, threat_level):
    features = np.array([[news_cnt, weather_dist, news_growth, threat_level]])
    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0]
    return pred, prob

# -------------------------------------------------------------
# 사이드바
# -------------------------------------------------------------
st.sidebar.title("🚢 모니터링 설정")
selected_route_key = st.sidebar.selectbox("상세 확인할 항로", list(ROUTE_DB.keys()))
ship_nationality = st.sidebar.selectbox("선박 국적 (기국)", ["한국", "미국", "중국", "영국", "이스라엘", "이란"])
ship_name = st.sidebar.text_input("선박명 / 식별부호", value="HANJIN GLORY")

st.sidebar.divider()
st.sidebar.subheader("위협 감지 파라미터 (선택 항로 기준)")
curr_route_data = ROUTE_DB[selected_route_key]
news_count = st.sidebar.slider("분쟁/이슈 뉴스 건수", 0, 60, curr_route_data["standard_risk_factors"]["news_cnt"])
news_growth = st.sidebar.slider("뉴스량 전일 대비 증가율 (%)", -50, 200, 40)
weather_distance = st.sidebar.slider("기상/위협 최근접 거리 (km)", 50, 1200, curr_route_data["standard_risk_factors"]["weather_dist_km"])
geopolitical_level = st.sidebar.select_slider("지정학적 위협 텐션 지수", options=[1,2,3,4,5], value=curr_route_data["standard_risk_factors"]["threat_level"])

alert_email = st.sidebar.text_input("비상 알림 수신 이메일", value="shipping_ops@trade.com")
send_alert_btn = st.sidebar.button("비상 알림 수동 발송")

st.sidebar.divider()
st.sidebar.subheader("실시간 AIS 데이터")
real_ship_data, selected_mmsi = None, None
if os.path.exists("ship_data.json"):
    with open("ship_data.json", "r", encoding="utf-8") as f:
        real_ship_data = json.load(f)
    st.sidebar.caption(f"업데이트: {real_ship_data['updated_at']}")
    ship_options = {m: (i.get("name","").strip() or f"MMSI:{m}") for m,i in real_ship_data["ships"].items() if "lat" in i}
    if ship_options:
        search_term = st.sidebar.text_input("선박명으로 검색 (예: PROTI)")
        if search_term:
            filtered = {k: v for k, v in ship_options.items() if search_term.upper() in v.upper()}
        else:
            filtered = ship_options
        if filtered:
            selected_mmsi = st.sidebar.selectbox("추적할 실제 선박", list(filtered.keys()), format_func=lambda x: filtered[x])
        else:
            st.sidebar.caption("검색 결과 없음")
            selected_mmsi = None
else:
    st.sidebar.caption("ship_data.json 없음 — collector.py 실행 필요")

# -------------------------------------------------------------
# 전체 항로 리스크 일괄 계산
# -------------------------------------------------------------
route_risk_summary = {}
for key, data in ROUTE_DB.items():
    if key == selected_route_key:
        n, w, g, t = news_count, weather_distance, news_growth, geopolitical_level
    else:
        f = data["standard_risk_factors"]
        n, w, g, t = f["news_cnt"], f["weather_dist_km"], 0, f["threat_level"]
    pred, prob = predict_risk(n, w, g, t)
    route_risk_summary[key] = {"pred": pred, "prob": prob, "rule": calc_rule_score(n, w, t)}

sel_pred = route_risk_summary[selected_route_key]["pred"]
sel_prob = route_risk_summary[selected_route_key]["prob"]
sel_rule = route_risk_summary[selected_route_key]["rule"]
labels = {0: "정상 (LOW)", 1: "경고 (MEDIUM)", 2: "심각 (HIGH)"}
risk_level_str = labels[sel_pred]
ALERT_TRIGGERED = sel_pred >= 1

curr_data = ROUTE_DB[selected_route_key]
scored_alternatives = []
for r in curr_data["alternatives"]:
    passage = r["eligibility"].get(ship_nationality, "확인 필요")
    penalty = 60 if any(k in passage for k in ["불가","표적","나포"]) else (20 if "조건부" in passage else 0)
    eff_safety = max(0, r["safety_score"] - penalty)
    cost_score = max(0, 100 - (r["cost_index_pct"]-100)*1.2)
    time_score = max(0, 100 - (r["transit_time_days"]-8)*2.0)
    total = round(eff_safety*0.5 + cost_score*0.3 + time_score*0.2, 1)
    item = dict(r); item["total_score"]=total; item["passage_status"]=passage
    scored_alternatives.append(item)
scored_alternatives.sort(key=lambda x: x["total_score"], reverse=True)
best_alt = scored_alternatives[0] if scored_alternatives else None

# -------------------------------------------------------------
# 메인 화면
# -------------------------------------------------------------
st.title("🌊 해상 항로 리스크 조기경보 시스템")
st.info(f"👈 사이드바에서 항로/파라미터를 조정하면 리스크가 실시간으로 변화합니다. 현재 선택: **{selected_route_key}**")

col1, col2, col3, col4 = st.columns(4)
col1.metric("선택 항로 리스크", risk_level_str)
col2.metric("규칙 기반 점수", f"{sel_rule} / 100")
col3.metric("AI 예측 위험 확률", f"{sel_prob[2]*100:.1f}%")
col4.metric("정상 항로 수", f"{sum(1 for v in route_risk_summary.values() if v['pred']==0)} / {len(ROUTE_DB)}")

st.divider()
st.subheader("🌍 전 세계 주요 항로 리스크 현황")

fig_world = go.Figure()
status_color = {0: "#22c55e", 1: "#eab308", 2: "#ef4444"}
status_label = {0: "정상", 1: "경고", 2: "심각"}

for key, data in ROUTE_DB.items():
    risk = route_risk_summary[key]
    tz = data["threat_zone"]
    is_selected = (key == selected_route_key)
    line_color = "#22d3ee" if is_selected else "#334155"
    path_lat = data["standard_path"]["lat"]
    path_lon = data["standard_path"]["lon"]

    # 글로우 바깥 레이어
    fig_world.add_trace(go.Scattergeo(
        lat=path_lat, lon=path_lon,
        mode="lines",
        line=dict(width=8 if is_selected else 3, color=line_color),
        opacity=0.15, showlegend=False, hoverinfo="skip"
    ))
    # 안쪽 선명한 선
    fig_world.add_trace(go.Scattergeo(
        lat=path_lat, lon=path_lon,
        mode="lines",
        line=dict(width=2.5 if is_selected else 1, color=line_color),
        showlegend=False, hoverinfo="skip"
    ))
    # 진행 방향 화살표 (선택된 항로만 표시해서 지도가 지저분해지지 않게)
    if is_selected:
        add_direction_arrows(fig_world, path_lat, path_lon, line_color, size=13, n_arrows=3)

    # 초크포인트 마커
    fig_world.add_trace(go.Scattergeo(
        lat=[tz["lat"]], lon=[tz["lon"]],
        mode="markers+text",
        marker=dict(size=26 if is_selected else 18, color=status_color[risk["pred"]],
                    line=dict(width=3 if is_selected else 1, color="white")),
        text=[key.split(" (")[0]], textposition="top center",
        textfont=dict(color="#f8fafc", size=11),
        name=f"{key} — {status_label[risk['pred']]}",
        hovertext=f"{key}<br>상태: {status_label[risk['pred']]}", hoverinfo="text"
    ))

fig_world.update_layout(
    geo=dict(projection_type="natural earth", showland=True, landcolor="rgb(30,41,59)",
        oceancolor="rgb(15,23,42)", showocean=True, showcoastlines=True, coastlinecolor="rgb(71,85,105)",
        showcountries=True, countrycolor="rgb(51,65,85)", bgcolor="rgba(0,0,0,0)"),
    paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=10,b=0), height=480,
    legend=dict(orientation="h", yanchor="bottom", y=-0.1, font=dict(color="#e2e8f0", size=11))
)
st.plotly_chart(fig_world, use_container_width=True)

# -------------------------------------------------------------
# 선택 항로 상세 지도
# -------------------------------------------------------------
st.divider()
st.subheader(f"🔍 상세 보기: {selected_route_key}")

ship_loc = curr_data["ship_location"]
threat_loc = curr_data["threat_zone"]
std_path = curr_data["standard_path"]

fig_detail = go.Figure()
fig_detail.add_trace(go.Scattergeo(
    lat=std_path["lat"], lon=std_path["lon"], mode="lines",
    line=dict(width=8, color="#3b82f6"), opacity=0.15, showlegend=False, hoverinfo="skip"
))
fig_detail.add_trace(go.Scattergeo(
    lat=std_path["lat"], lon=std_path["lon"], mode="lines",
    line=dict(width=3, color="#3b82f6"), name="표준 항로"
))
# 표준 항로 진행 방향 화살표
add_direction_arrows(fig_detail, std_path["lat"], std_path["lon"], "#3b82f6", size=14, n_arrows=4)

threat_color = "#ef4444" if ALERT_TRIGGERED else "#64748b"
fig_detail.add_trace(go.Scattergeo(
    lat=[threat_loc["lat"]], lon=[threat_loc["lon"]], mode="markers+text",
    marker=dict(size=22, color=threat_color), text=[threat_loc["name"]],
    textposition="top center", textfont=dict(color="#f8fafc"),
    name="⚠ 위협 감지 구역" if ALERT_TRIGGERED else "모니터링 구역"
))
fig_detail.add_trace(go.Scattergeo(
    lat=[ship_loc["lat"]], lon=[ship_loc["lon"]], mode="markers+text",
    marker=dict(size=16, color="#ffc107", symbol="triangle-up"), text=[ship_name],
    textposition="bottom center", textfont=dict(color="#f8fafc"), name="시나리오 선박"
))
if ALERT_TRIGGERED and best_alt:
    fig_detail.add_trace(go.Scattergeo(
        lat=best_alt["path_lat"], lon=best_alt["path_lon"], mode="lines",
        line=dict(width=4, color="#dc3545", dash="dash"), name=f"추천 대체: {best_alt['route_name']}"
    ))
    # 대체 루트 진행 방향 화살표
    add_direction_arrows(fig_detail, best_alt["path_lat"], best_alt["path_lon"], "#dc3545", size=14, n_arrows=4)

if real_ship_data:
    other_lat, other_lon, other_name = [], [], []
    my_lat = my_lon = my_name = None
    for m, info in real_ship_data["ships"].items():
        if "lat" not in info: continue
        if m == selected_mmsi:
            my_lat, my_lon = info["lat"], info["lon"]
            my_name = info.get("name","").strip() or f"MMSI:{m}"
        else:
            other_lat.append(info["lat"]); other_lon.append(info["lon"])
            other_name.append(info.get("name","").strip() or f"MMSI:{m}")
    if other_lat:
        fig_detail.add_trace(go.Scattergeo(lat=other_lat, lon=other_lon, mode="markers",
            marker=dict(size=4, color="#94a3b8", opacity=0.5), text=other_name, hoverinfo="text",
            name="실시간 다른 선박"))
    if my_lat:
        fig_detail.add_trace(go.Scattergeo(lat=[my_lat], lon=[my_lon], mode="markers+text",
            marker=dict(size=18, color="red", symbol="star", line=dict(width=2, color="white")),
            text=[my_name], textposition="top center", textfont=dict(color="#f8fafc"), name="내 지정 선박"))

fig_detail.update_layout(
    geo=dict(projection_type="equirectangular", showland=True, landcolor="rgb(30,41,59)",
        oceancolor="rgb(15,23,42)", showocean=True, showcoastlines=True, coastlinecolor="rgb(71,85,105)",
        showcountries=True, countrycolor="rgb(51,65,85)",
        center=dict(lat=ship_loc["lat"], lon=ship_loc["lon"]), projection_scale=2.2, bgcolor="rgba(0,0,0,0)"),
    paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=10,b=0), height=550,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(color="#e2e8f0"))
)
st.plotly_chart(fig_detail, use_container_width=True)

# -------------------------------------------------------------
# 상황판단 + 뉴스
# -------------------------------------------------------------
st.divider()
col_status, col_news = st.columns([1.3, 1])

with col_status:
    st.subheader("현재 상황 판단")
    if ALERT_TRIGGERED:
        if sel_pred == 2:
            st.error(f"🔴 심각 등급 — {threat_loc['name']} 인근 위협 감지")
        else:
            st.warning(f"🟡 경고 등급 — {threat_loc['name']} 인근 리스크 상승")
        st.markdown(f"""
**우려 요인**
- 관련 뉴스: **{news_count}건** (전일 대비 {news_growth:+d}%)
- 위협 최근접 거리: **{weather_distance}km**
- 지정학적 텐션: **{geopolitical_level}/5**

**권장 조치**: **[{best_alt['route_name']}]** 전환 검토
""")
    else:
        st.success("🟢 정상 운항 상태입니다.")

with col_news:
    st.subheader("관련 뉴스")
    news_items = fetch_news(curr_data["news_keyword"])
    if news_items is None:
        st.caption("NewsAPI 키 미설정")
    elif not news_items:
        st.caption("관련 뉴스 없음")
    else:
        for n in news_items:
            st.markdown(f"**[{n['title']}]({n['link']})**")
            st.caption((n["desc"] or "")[:90] + "...")

# -------------------------------------------------------------
# 리스크 점수 추이 (30일)
# -------------------------------------------------------------
st.divider()
st.subheader("📈 리스크 점수 추이 (최근 30일)")

np.random.seed(hash(selected_route_key) % 1000)
trend_base = sel_rule
trend_values = np.clip(trend_base + np.cumsum(np.random.randn(30) * 4), 0, 100)

fig_trend = go.Figure()
fig_trend.add_trace(go.Scatter(
    y=trend_values, mode="lines",
    line=dict(color="#f59e0b", width=2),
    fill="tozeroy", fillcolor="rgba(245, 158, 11, 0.15)",
    name="리스크 점수"
))
fig_trend.update_layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#94a3b8"),
    height=220, margin=dict(l=10, r=10, t=10, b=10),
    xaxis=dict(showgrid=False, title="일 전"),
    yaxis=dict(showgrid=True, gridcolor="#1e293b", title="점수"),
    showlegend=False
)
st.plotly_chart(fig_trend, use_container_width=True)

# -------------------------------------------------------------
# 대체 루트 (경보시만)
# -------------------------------------------------------------
if ALERT_TRIGGERED:
    st.divider()
    st.subheader("대체 루트 추천")
    for idx, r in enumerate(scored_alternatives):
        badge = "🟢" if idx == 0 else "🔵"
        with st.container():
            st.markdown(f"### {badge} {idx+1}순위: {r['route_name']}")
            c1,c2,c3,c4,c5 = st.columns(5)
            c1.metric("점수", f"{r['total_score']}")
            c2.metric("소요시간", f"{r['transit_time_days']}일")
            c3.metric("운임지수", f"{r['cost_index_pct']}%")
            c4.metric("보험료", r["war_risk_insurance"])
            c5.metric(f"통행({ship_nationality})", r["passage_status"])
            st.info(r["recommendation_reason"])

            risk_tag, badge_color = ("🟢 Low", "#14532d") if r["total_score"] >= 70 else \
                (("🟡 Medium", "#78350f") if r["total_score"] >= 40 else ("🔴 High", "#7f1d1d"))
            st.markdown(
                f"<span class='risk-badge' style='background:{badge_color};color:white;'>{risk_tag} Risk</span>",
                unsafe_allow_html=True
            )

    if send_alert_btn:
        st.sidebar.success(f"📧 발송됨: {alert_email}")