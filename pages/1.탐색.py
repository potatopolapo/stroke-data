import streamlit as st
import pandas as pd
import plotly.express as px

# ---------------------------------------------------
# 1. 페이지 기본 설정
# ---------------------------------------------------
st.set_page_config(
    page_title="뇌졸중 예측 실습실 - 탐색",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 데이터 탐색")
st.write("뇌졸중 데이터의 다양한 특징을 그래프와 표로 살펴봅니다.")

st.divider()

# ---------------------------------------------------
# 2. 데이터 불러오기 (첫 화면과 동일)
# ---------------------------------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
    df = pd.read_csv(url, encoding="utf-8")
    return df

df = load_data()

# ---------------------------------------------------
# 3. 나이와 평균 혈당의 분포 - 히스토그램 두 개 나란히
# ---------------------------------------------------
st.subheader("1️⃣ 나이와 평균 혈당의 분포")

hist_col1, hist_col2 = st.columns(2)

with hist_col1:
    fig_age_hist = px.histogram(
        df, x="age",
        nbins=30,
        title="나이(age) 분포",
        labels={"age": "나이"}
    )
    st.plotly_chart(fig_age_hist, use_container_width=True)

with hist_col2:
    fig_glucose_hist = px.histogram(
        df, x="avg_glucose_level",
        nbins=30,
        title="평균 혈당(avg_glucose_level) 분포",
        labels={"avg_glucose_level": "평균 혈당"}
    )
    st.plotly_chart(fig_glucose_hist, use_container_width=True)

st.divider()

# ---------------------------------------------------
# 4. 뇌졸중 여부에 따른 나이·평균 혈당 상자그림 + 평균값 표
# ---------------------------------------------------
st.subheader("2️⃣ 뇌졸중 여부에 따른 나이·평균 혈당 비교")

# stroke 값(0, 1)을 한글로 바꾼 열을 새로 만들어서 그래프에 사용
df_box = df.copy()
df_box["뇌졸중 여부"] = df_box["stroke"].map({0: "뇌졸중 없음", 1: "뇌졸중 있음"})

box_col1, box_col2 = st.columns(2)

with box_col1:
    fig_age_box = px.box(
        df_box, x="뇌졸중 여부", y="age",
        color="뇌졸중 여부",
        title="뇌졸중 여부별 나이 분포",
        labels={"age": "나이"}
    )
    st.plotly_chart(fig_age_box, use_container_width=True)

with box_col2:
    fig_glucose_box = px.box(
        df_box, x="뇌졸중 여부", y="avg_glucose_level",
        color="뇌졸중 여부",
        title="뇌졸중 여부별 평균 혈당 분포",
        labels={"avg_glucose_level": "평균 혈당"}
    )
    st.plotly_chart(fig_glucose_box, use_container_width=True)

# 두 그룹의 평균값 표
mean_table = df_box.groupby("뇌졸중 여부")[["age", "avg_glucose_level"]].mean().reset_index()
mean_table.columns = ["뇌졸중 여부", "나이 평균", "평균 혈당 평균"]
mean_table["나이 평균"] = mean_table["나이 평균"].round(2)
mean_table["평균 혈당 평균"] = mean_table["평균 혈당 평균"].round(2)

st.write("**그룹별 평균값**")
st.dataframe(mean_table, use_container_width=True, hide_index=True)

st.divider()

# ---------------------------------------------------
# 5. 고혈압·심장병 유무에 따른 뇌졸중 비율 막대그래프
# ---------------------------------------------------
st.subheader("3️⃣ 고혈압·심장병 유무에 따른 뇌졸중 비율")

bar_col1, bar_col2 = st.columns(2)

with bar_col1:
    # 고혈압 유무별 뇌졸중 비율(%) 계산
    hyper_ratio = df.groupby("hypertension")["stroke"].mean().reset_index()
    hyper_ratio["stroke"] = hyper_ratio["stroke"] * 100
    hyper_ratio["hypertension"] = hyper_ratio["hypertension"].map({0: "고혈압 없음", 1: "고혈압 있음"})

    fig_hyper = px.bar(
        hyper_ratio, x="hypertension", y="stroke",
        title="고혈압 유무에 따른 뇌졸중 비율",
        labels={"hypertension": "고혈압 유무", "stroke": "뇌졸중 비율(%)"},
        text_auto=".2f"
    )
    st.plotly_chart(fig_hyper, use_container_width=True)

with bar_col2:
    # 심장병 유무별 뇌졸중 비율(%) 계산
    heart_ratio = df.groupby("heart_disease")["stroke"].mean().reset_index()
    heart_ratio["stroke"] = heart_ratio["stroke"] * 100
    heart_ratio["heart_disease"] = heart_ratio["heart_disease"].map({0: "심장병 없음", 1: "심장병 있음"})

    fig_heart = px.bar(
        heart_ratio, x="heart_disease", y="stroke",
        title="심장병 유무에 따른 뇌졸중 비율",
        labels={"heart_disease": "심장병 유무", "stroke": "뇌졸중 비율(%)"},
        text_auto=".2f"
    )
    st.plotly_chart(fig_heart, use_container_width=True)

st.divider()

# ---------------------------------------------------
# 6. bmi 결측치 그룹의 뇌졸중 비율 vs 전체 뇌졸중 비율
# ---------------------------------------------------
st.subheader("4️⃣ 체질량지수(bmi) 결측 여부에 따른 뇌졸중 비율")

bmi_missing_count = df["bmi"].isnull().sum()
bmi_missing_stroke_ratio = df[df["bmi"].isnull()]["stroke"].mean() * 100
overall_stroke_ratio = df["stroke"].mean() * 100

bmi_compare_table = pd.DataFrame({
    "구분": ["bmi 결측자", "전체"],
    "사람 수": [bmi_missing_count, len(df)],
    "뇌졸중 비율(%)": [round(bmi_missing_stroke_ratio, 2), round(overall_stroke_ratio, 2)]
})

st.dataframe(bmi_compare_table, use_container_width=True, hide_index=True)

st.divider()

# ---------------------------------------------------
# 7. 흡연 상태별 사람 수 표
# ---------------------------------------------------
st.subheader("5️⃣ 흡연 상태(smoking_status)별 사람 수")

smoking_count = df["smoking_status"].value_counts().reset_index()
smoking_count.columns = ["흡연 상태", "사람 수"]

st.dataframe(smoking_count, use_container_width=True, hide_index=True)
