import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier

# -------------------------------------------------------------
# 1. 기본 설정 및 데이터 정의 (정밀 바닷길 + 추천 로직 속성)
# -------------------------------------------------------------
st.set_page_config(
    page_title="해상 항로 리스크 조기경보 및 대체 루트 추천 시스템",
    layout="wide"
)

ROUTE_DB = {
    "아시아 - 미주 동안 (파나마 운하)": {
        "standard_risk_factors": {"news_cnt": 15, "weather_dist_km": 300, "threat_level": 3},
        "items": ["전자제품(HS 8542)", "자동차부품(HS 8708)", "소비재(HS 9503)"],
        "ship_location": {"lat": 9.38, "lon": -79.92, "name": "파나마 운하 대기 구역(콜론)"},
        "threat_zone": {"lat": 9.08, "lon": -79.68, "name": "가툰 호수 가뭄/통항 제한 구역"},
        "alternatives": [
            {
                "route_name": "기존 항로: 파나마 운하 통과",
                "transit_time_days": 24,
                "cost_index_pct": 100,
                "safety_score": 50,  # 가뭄 대기 페널티
                "war_risk_insurance": "일반 요율 (가뭄 할증 +$10,000/선박)",
                "eligibility": {"한국": "통행 가능", "미국": "통행 가능", "중국": "통행 가능", "영국": "통행 가능", "이스라엘": "통행 가능"},
                "status": "가뭄 대기지연",
                "recommendation_reason": "가장 짧은 기본 항로이나 가툰 호수 저수위로 인해 사전 예약 슬롯 미확보 시 통항 대기가 7~10일 발생할 수 있습니다.",
                "path_lat": [35.1, 34.0, 30.0, 20.0, 9.38, 9.12, 9.35, 15.0, 24.0, 32.0, 40.7],
                "path_lon": [129.0, 145.0, 180.0, -140.0, -79.92, -79.7, -79.9, -75.0, -74.0, -75.0, -74.0]
            },
            {
                "route_name": "우회 항로: 수에즈 운하 경유 (역방향)",
                "transit_time_days": 34,
                "cost_index_pct": 130,
                "safety_score": 80,
                "war_risk_insurance": "할증 가능",
                "eligibility": {"한국": "통행 가능", "미국": "조건부", "중국": "통행 가능", "영국": "조건부", "이스라엘": "통행 불가"},
                "status": "추천 우회로",
                "recommendation_reason": "대기 지연 없이 정시 도착을 보장하며 대형 컨테이너선 운항에 가장 안정적인 우회 옵션입니다.",
                "path_lat": [35.1, 23.0, 1.3, 6.0, 12.5, 27.5, 31.0, 36.0, 36.0, 38.0, 40.7],
                "path_lon": [129.0, 120.0, 103.8, 80.0, 43.5, 34.0, 32.3, 15.0, -5.6, -40.0, -74.0]
            },
            {
                "route_name": "우회 항로: 남아메리카 마젤란/혼곶 우회",
                "transit_time_days": 42,
                "cost_index_pct": 155,
                "safety_score": 65,
                "war_risk_insurance": "일반 요율",
                "eligibility": {"한국": "통행 가능", "미국": "통행 가능", "중국": "통행 가능", "영국": "통행 가능", "이스라엘": "통행 가능"},
                "status": "초대형선(비운하) 대체안",
                "recommendation_reason": "운하 통행료는 없으나 항해 일수가 과도하게 증가(42일)하고 남극해 인접 기상 리스크가 존재합니다.",
                "path_lat": [35.1, 20.0, -10.0, -35.0, -52.0, -56.0, -40.0, -10.0, 20.0, 40.7],
                "path_lon": [129.0, 140.0, -170.0, -120.0, -75.0, -67.0, -50.0, -35.0, -60.0, -74.0]
            }
        ]
    },
    "아시아 - 유럽 (홍해 / 수에즈 운하)": {
        "standard_risk_factors": {"news_cnt": 28, "weather_dist_km": 800, "threat_level": 4},
        "items": ["자동차/부품(HS 8708)", "이차전지(HS 8507)", "전자부품(HS 8542)"],
        "ship_location": {"lat": 12.8, "lon": 44.5, "name": "아덴만 진입부"},
        "threat_zone": {"lat": 14.5, "lon": 42.5, "name": "홍해 남부 분쟁 위험 구역"},
        "alternatives": [
            {
                "route_name": "기존 항로: 수에즈 운하 직통",
                "transit_time_days": 25,
                "cost_index_pct": 100,
                "safety_score": 20,  # 분쟁 고위험
                "war_risk_insurance": "할증 적용 (+250%)",
                "eligibility": {"한국": "조건부 가능", "미국": "표적 위험", "영국": "표적 위험", "중국": "통행 가능", "이스라엘": "통행 불가"},
                "status": "고위험",
                "recommendation_reason": "소요 시간은 가장 짧으나 홍해 군사 위협 및 전쟁보험료 폭증(+250%)으로 선박 안전상 위험도가 매우 높습니다.",
                "path_lat": [35.1, 22.0, 1.3, 5.8, 12.5, 20.0, 31.0, 36.5, 36.0, 44.0, 51.9],
                "path_lon": [129.0, 120.0, 103.8, 80.5, 44.0, 38.5, 32.3, 15.0, -5.6, -9.5, 4.3]
            },
            {
                "route_name": "우회 항로: 아프리카 희망봉",
                "transit_time_days": 38,
                "cost_index_pct": 142,
                "safety_score": 95,
                "war_risk_insurance": "일반 요율",
                "eligibility": {"한국": "통행 가능", "미국": "통행 가능", "영국": "통행 가능", "중국": "통행 가능", "이스라엘": "통행 가능"},
                "status": "최우선 권장",
                "recommendation_reason": "운항 일수가 약 13일 추가되지만 피격 위험 0% 및 전쟁보험료 면제로 현재 가장 권장되는 글로벌 표준 우회로입니다.",
                "path_lat": [35.1, 22.0, 1.3, -5.0, -20.0, -34.8, -34.5, 0.0, 20.0, 44.0, 51.9],
                "path_lon": [129.0, 120.0, 103.8, 90.0, 60.0, 20.0, 18.0, -10.0, -18.0, -9.5, 4.3]
            }
        ]
    },
    "중동 - 동아시아 (호르무즈 해협)": {
        "standard_risk_factors": {"news_cnt": 18, "weather_dist_km": 500, "threat_level": 4},
        "items": ["원유(HS 2709)", "석유제품(HS 2710)", "LPG(HS 2711)"],
        "ship_location": {"lat": 26.5, "lon": 56.5, "name": "호르무즈 해협 진입부"},
        "threat_zone": {"lat": 26.8, "lon": 55.8, "name": "호르무즈 북부 군사 긴장 구역"},
        "alternatives": [
            {
                "route_name": "기존 항로: 호르무즈 해협 통과",
                "transit_time_days": 15,
                "cost_index_pct": 100,
                "safety_score": 35,
                "war_risk_insurance": "할증 적용 (+300%)",
                "eligibility": {"한국": "통행 가능", "미국": "공격 위험", "영국": "공격 위험", "중국": "통행 가능", "이스라엘": "나포 위험"},
                "status": "위험",
                "recommendation_reason": "원유 수송 주력 루트이나 이란 해역 인접 군사 긴장으로 인해 지정학적 나포 및 공격 위험이 상존합니다.",
                "path_lat": [27.0, 26.5, 24.5, 15.0, 5.8, 1.3, 12.0, 22.0, 35.1],
                "path_lon": [50.2, 56.5, 59.0, 68.0, 80.5, 103.8, 112.0, 120.0, 129.0]
            },
            {
                "route_name": "대체 항로: 사우디 서안 얀부항(홍해) 선적 연계",
                "transit_time_days": 19,
                "cost_index_pct": 135,
                "safety_score": 85,
                "war_risk_insurance": "일반 요율",
                "eligibility": {"한국": "통행 가능", "미국": "통행 가능", "영국": "통행 가능", "중국": "통행 가능", "이스라엘": "조건부"},
                "status": "파이프라인 연계안",
                "recommendation_reason": "호르무즈 해협을 완전 우회하여 동서 파이프라인 육상 수송 후 선적하므로 해협 봉쇄 시 유일한 현실적 대체 공급망입니다.",
                "path_lat": [24.1, 15.0, 12.5, 10.0, 5.8, 1.3, 12.0, 22.0, 35.1],
                "path_lon": [38.0, 42.0, 44.0, 60.0, 80.5, 103.8, 112.0, 120.0, 129.0]
            }
        ]
    },
    "동남아 - 동아시아 (말라카 해협)": {
        "standard_risk_factors": {"news_cnt": 4, "weather_dist_km": 120, "threat_level": 2},
        "items": ["반도체(HS 8542)", "정밀기계(HS 8479)", "석유화학(HS 2901)"],
        "ship_location": {"lat": 2.5, "lon": 101.8, "name": "말라카 해협 중앙"},
        "threat_zone": {"lat": 4.0, "lon": 100.5, "name": "열대성 폭풍 및 해적 빈발 구역"},
        "alternatives": [
            {
                "route_name": "기존 항로: 말라카 해협 통과",
                "transit_time_days": 8,
                "cost_index_pct": 100,
                "safety_score": 75,
                "war_risk_insurance": "일반 요율",
                "eligibility": {"한국": "통행 가능", "미국": "통행 가능", "중국": "통행 가능", "영국": "통행 가능", "이스라엘": "통행 가능"},
                "status": "기상 주의",
                "recommendation_reason": "비용과 시간 면에서 압도적으로 유리하므로 태풍 직접 통과 예보가 아닐 경우 감속 운항 유지가 권장됩니다.",
                "path_lat": [1.3, 3.0, 5.5, 12.0, 22.0, 35.1],
                "path_lon": [103.8, 101.0, 98.0, 112.0, 120.0, 129.0]
            },
            {
                "route_name": "우회 항로: 순다 / 롬복 해협 경유",
                "transit_time_days": 12,
                "cost_index_pct": 125,
                "safety_score": 90,
                "war_risk_insurance": "일반 요율",
                "eligibility": {"한국": "통행 가능", "미국": "통행 가능", "중국": "통행 가능", "영국": "통행 가능", "이스라엘": "통행 가능"},
                "status": "기상 회피안",
                "recommendation_reason": "말라카 해협 내 초강력 태풍/폭풍 발생 시 동인도네시아 심해 수로를 통해 안전하게 항해할 수 있습니다.",
                "path_lat": [-6.0, -8.7, -3.0, 5.0, 18.0, 35.1],
                "path_lon": [105.5, 115.8, 118.5, 124.0, 128.0, 129.0]
            }
        ]
    }
}

# -------------------------------------------------------------
# 2. 리스크 머신러닝 모델
# -------------------------------------------------------------
@st.cache_resource
def get_trained_risk_model():
    np.random.seed(42)
    X = np.random.rand(400, 4)
    X[:, 0] = X[:, 0] * 50
    X[:, 1] = X[:, 1] * 1000
    X[:, 2] = (X[:, 2] - 0.5) * 200
    X[:, 3] = np.random.randint(1, 6, 400)

    risk_score = (X[:, 0] * 0.3) + ((1000 - X[:, 1]) * 0.05) + (X[:, 2] * 0.2) + (X[:, 3] * 10)
    y = np.where(risk_score > 65, 2, np.where(risk_score > 35, 1, 0))

    model = RandomForestClassifier(n_estimators=30, random_state=42)
    model.fit(X, y)
    return model

model = get_trained_risk_model()

# -------------------------------------------------------------
# 3. 사이드바 설정
# -------------------------------------------------------------
st.sidebar.title("1. 모니터링 항로 설정")
selected_route_key = st.sidebar.selectbox("대상 주요 글로벌 항로", list(ROUTE_DB.keys()))
ship_nationality = st.sidebar.selectbox("선박 국적 (기국)", ["한국", "미국", "중국", "영국", "이스라엘", "이란"])
ship_name = st.sidebar.text_input("선박명 / 식별부호", value="GLOBAL_CARRIER_01")

st.sidebar.divider()
st.sidebar.title("2. 위협 감지 파라미터")
curr_route_data = ROUTE_DB[selected_route_key]
news_count = st.sidebar.slider("분쟁/이슈 뉴스 건수", 0, 60, curr_route_data["standard_risk_factors"]["news_cnt"])
news_growth = st.sidebar.slider("뉴스량 전일 대비 증가율 (%)", -50, 200, 40)
weather_distance = st.sidebar.slider("기상/위협 최근접 거리 (km)", 50, 1200, curr_route_data["standard_risk_factors"]["weather_dist_km"])
geopolitical_level = st.sidebar.select_slider("지정학적 위협 텐션 지수", options=[1, 2, 3, 4, 5], value=curr_route_data["standard_risk_factors"]["threat_level"])

alert_email = st.sidebar.text_input("비상 알림 수신 이메일", value="shipping_ops@trade.com")
send_alert_btn = st.sidebar.button("비상 알림 수동 발송")

# -------------------------------------------------------------
# 4. 리스크 판정 및 추천 순위 산출 로직
# -------------------------------------------------------------
rule_score = min(100, int((news_count * 1.2) + max(0, (600 - weather_distance) * 0.08) + (geopolitical_level * 10)))
features = np.array([[news_count, weather_distance, news_growth, geopolitical_level]])
ml_pred = model.predict(features)[0]
ml_prob = model.predict_proba(features)[0]

labels = {0: "정상 (LOW)", 1: "경고 (MEDIUM)", 2: "심각 (HIGH)"}
risk_level_str = labels[ml_pred]

# 종합 추천 지수 계산 (안전성 45%, 비용 35%, 시간 20% 반영)
scored_alternatives = []
for r in curr_route_data["alternatives"]:
    # 통행 불가/위험 국적인 경우 안전성 감점
    passage = r["eligibility"].get(ship_nationality, "확인 필요")
    eligibility_penalty = 0
    if "불가" in passage or "표적" in passage or "나포" in passage:
        eligibility_penalty = 60
    elif "조건부" in passage:
        eligibility_penalty = 20

    effective_safety = max(0, r["safety_score"] - eligibility_penalty)
    
    # 종합 점수 (100점 만점 기준 환산)
    # 비용지수 낮을수록, 소요시간 적을수록, 안전성 높을수록 점수 상승
    cost_score = max(0, 100 - (r["cost_index_pct"] - 100) * 1.2)
    time_score = max(0, 100 - (r["transit_time_days"] - 8) * 2.0)
    total_recommend_score = (effective_safety * 0.50) + (cost_score * 0.30) + (time_score * 0.20)
    
    item_copy = dict(r)
    item_copy["total_score"] = round(total_recommend_score, 1)
    item_copy["effective_safety"] = effective_safety
    item_copy["passage_status"] = passage
    scored_alternatives.append(item_copy)

# 추천 점수 높은 순으로 정렬
scored_alternatives = sorted(scored_alternatives, key=lambda x: x["total_score"], reverse=True)

# -------------------------------------------------------------
# 5. 메인 대시보드 레이아웃
# -------------------------------------------------------------
st.title("해상 항로 리스크 조기경보 및 대체 루트 추천 시스템")
st.caption(f"선박: {ship_name} | 국적: {ship_nationality} | 선택 항로: {selected_route_key}")

col1, col2, col3, col4 = st.columns(4)
col1.metric("최종 리스크 등급", risk_level_str)
col2.metric("규칙 기반 환산 점수", f"{rule_score} / 100")
col3.metric("AI 예측 위험 확률", f"{ml_prob[2]*100:.1f}%")
col4.metric("위협요인 최근접 거리", f"{weather_distance} km")

st.divider()

# 바닷길 항로 지도 시각화
st.subheader("실시간 선박 위치 및 추천 항로(바닷길) 맵")

ship_loc = curr_route_data["ship_location"]
threat_loc = curr_route_data["threat_zone"]

fig_map = go.Figure()
rank_colors = ["#28a745", "#007bff", "#6f42c1", "#dc3545"]

for rank_idx, r in enumerate(scored_alternatives):
    color = rank_colors[rank_idx % len(rank_colors)]
    rank_label = f"[{rank_idx + 1}순위 추천] {r['route_name']}"
    fig_map.add_trace(go.Scattergeo(
        lat=r["path_lat"],
        lon=r["path_lon"],
        mode="lines+markers",
        line=dict(width=3 if rank_idx == 0 else 2, color=color, dash="solid" if rank_idx == 0 else "dash"),
        marker=dict(size=4),
        name=rank_label,
        hoverinfo="text",
        text=[f"{rank_label}<br>소요: {r['transit_time_days']}일 | 추천점수: {r['total_score']}점" for _ in r["path_lat"]]
    ))

# 위험 구역 마커
fig_map.add_trace(go.Scattergeo(
    lat=[threat_loc["lat"]],
    lon=[threat_loc["lon"]],
    mode="markers+text",
    marker=dict(size=24, color="rgba(220, 53, 69, 0.75)", symbol="circle"),
    text=[threat_loc["name"]],
    textposition="top center",
    name="위험/통제 구역"
))

# 현재 선박 위치 마커
fig_map.add_trace(go.Scattergeo(
    lat=[ship_loc["lat"]],
    lon=[ship_loc["lon"]],
    mode="markers+text",
    marker=dict(size=16, color="#ffc107", symbol="triangle-up"),
    text=[f"현재 위치: {ship_name}"],
    textposition="bottom center",
    name="선박 현재 위치"
))

fig_map.update_layout(
    geo=dict(
        projection_type="equirectangular",
        showland=True,
        landcolor="rgb(235, 235, 235)",
        oceancolor="rgb(215, 232, 248)",
        showocean=True,
        showcoastlines=True,
        coastlinecolor="rgb(160, 160, 160)",
        showcountries=True,
        countrycolor="rgb(200, 200, 200)",
        center=dict(lat=ship_loc["lat"], lon=ship_loc["lon"]),
        projection_scale=1.5
    ),
    margin=dict(l=0, r=0, t=30, b=0),
    height=500,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig_map, use_container_width=True)

st.divider()

# -------------------------------------------------------------
# 6. 추천 순위별 상세 비교 및 추천 이유 섹션
# -------------------------------------------------------------
st.subheader("대체 루트 추천 순위별 종합 비교 및 사유 분석")

# 순위별 카드 렌더링
for rank_idx, r in enumerate(scored_alternatives):
    rank_num = rank_idx + 1
    with st.container():
        badge_color = "🟢" if rank_num == 1 else ("🔵" if rank_num == 2 else "⚪")
        st.markdown(f"### {badge_color} **{rank_num}순위 추천**: {r['route_name']}")
        
        c1, c2, c3, c4, c5 = st.columns([1.5, 1.2, 1.2, 1.5, 1.5])
        c1.metric("종합 추천 점수", f"{r['total_score']} 점")
        c2.metric("예상 소요 시간", f"{r['transit_time_days']} 일")
        c3.metric("운임 지수", f"{r['cost_index_pct']} % (기준=100)")
        c4.metric("전쟁보험료", r["war_risk_insurance"])
        c5.metric(f"통행 가능 여부({ship_nationality})", r["passage_status"])
        
        st.info(f"💡 **추천 / 평가 사유**: {r['recommendation_reason']}")
        st.write("")

# 순위표 데이터프레임 요약
comp_summary = []
for rank_idx, r in enumerate(scored_alternatives):
    comp_summary.append({
        "추천순위": f"{rank_idx + 1}순위",
        "항로명": r["route_name"],
        "종합점수": r["total_score"],
        "소요시간(일)": r["transit_time_days"],
        "운임비용지수(%)": r["cost_index_pct"],
        f"통행판정({ship_nationality})": r["passage_status"],
        "핵심 추천 사유": r["recommendation_reason"]
    })

st.dataframe(pd.DataFrame(comp_summary), use_container_width=True)

# 차트 비교
chart_df = pd.DataFrame({
    "항로 (추천순)": [f"[{i+1}순위] " + r["route_name"] for i, r in enumerate(scored_alternatives)],
    "종합추천점수": [r["total_score"] for r in scored_alternatives],
    "운항소요시간(일)": [r["transit_time_days"] for r in scored_alternatives],
    "운임비용지수": [r["cost_index_pct"] for r in scored_alternatives]
})

fig_bar = px.bar(
    chart_df,
    x="항로 (추천순)",
    y=["종합추천점수", "운항소요시간(일)", "운임비용지수"],
    barmode="group",
    title="추천 순위별 지표 비교 (종합점수 vs 소요시간 vs 비용지수)"
)
st.plotly_chart(fig_bar, use_container_width=True)

# -------------------------------------------------------------
# 7. 실무 액션 체크리스트
# -------------------------------------------------------------
st.divider()
st.subheader("상황별 실무 대응 체크리스트")

best_route = scored_alternatives[0]["route_name"]

if ml_pred == 2:
    st.error(f"주의: 고위험 등급 판정. 즉시 최우선 추천 루트인 [{best_route}] 전환 검토를 권장합니다.")
    st.checkbox("1. 1순위 추천 항로 선사 및 포워더와 선복 계약 전환 협의")
    st.checkbox("2. 파나마/홍해 통항 불가 및 대기 지연에 따른 화주 납기 조정 통보")
    st.checkbox(f"3. {ship_nationality} 국적 선박 표적 가능성 점검 및 선박 보안 레벨 상향")
elif ml_pred == 1:
    st.warning(f"경고: 리스크 주의 단계. 현 항로 유지 시 대기 시간 추이와 [{best_route}] 운임 변동을 비교하십시오.")
    st.checkbox("1. 운하/해협 대기 일수 및 통항 슬롯 실시간 확인")
    st.checkbox("2. 우회 항로 벙커유(연료) 추가 소모량 및 비용 사전 산출")
else:
    st.success("안전: 정상 운항 상태입니다.")
    st.checkbox("1. 표준 항해 일정 및 AIS 정상 보고 유지")

if send_alert_btn:
    st.sidebar.success(f"[알림 발송 완료] 수신처: {alert_email} | 등급: {risk_level_str} | 최우선 추천: {best_route}")