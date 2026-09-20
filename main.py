import streamlit as st
import pandas as pd

# ---------------------------------------------------
# 1. 페이지 기본 설정
#    - page_title: 브라우저 탭에 표시되는 제목
#    - page_icon: 브라우저 탭에 표시되는 아이콘(이모지 가능)
#    - layout: 화면을 넓게 쓰기 위해 wide로 설정
# ---------------------------------------------------
st.set_page_config(
    page_title="뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide"
)

# ---------------------------------------------------
# 2. 화면 맨 위 제목 (아이콘 포함)
# ---------------------------------------------------
st.title("🧠 뇌졸중 예측 실습실")
st.write("뇌졸중 데이터를 살펴보고, 예측 모델을 함께 만들어보는 실습 공간입니다.")

st.divider()

# ---------------------------------------------------
# 3. 데이터 불러오기
#    - @st.cache_data: 같은 데이터를 반복해서 다운로드하지 않도록
#      한 번 불러온 데이터를 저장(캐시)해두는 기능
# ---------------------------------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
    df = pd.read_csv(url, encoding="utf-8")
    return df

df = load_data()

# ---------------------------------------------------
# 4. 큰 숫자 카드 네 개 (전체 인원, 열 개수, 뇌졸중 환자 수, 비율)
#    - st.columns(4): 화면을 4칸으로 나누기
#    - st.metric: 큰 숫자 카드를 보여주는 위젯
# ---------------------------------------------------
st.subheader("📊 데이터 한눈에 보기")

total_count = len(df)                     # 전체 사람 수 (행 개수)
col_count = df.shape[1]                   # 열 개수
stroke_count = int(df["stroke"].sum())    # stroke가 1인 사람 수
stroke_ratio = stroke_count / total_count * 100  # 비율(%)

card1, card2, card3, card4 = st.columns(4)

with card1:
    st.metric(label="전체 사람 수", value=f"{total_count:,} 명")

with card2:
    st.metric(label="열 개수", value=f"{col_count} 개")

with card3:
    st.metric(label="뇌졸중 발생자 수", value=f"{stroke_count:,} 명")

with card4:
    st.metric(label="뇌졸중 발생 비율", value=f"{stroke_ratio:.2f} %")

st.divider()

# ---------------------------------------------------
# 5. 열 정보 표
#    - 열 이름, 우리말 뜻(직접 채워 넣을 빈 칸), 값의 종류, 빈 값 개수
#    - data_editor를 사용하면 표를 직접 화면에서 수정할 수 있음
# ---------------------------------------------------
st.subheader("📋 열(컬럼) 정보")

# 각 열의 값 종류를 문자열로 정리하는 함수
def get_value_types(series):
    unique_vals = series.dropna().unique()
    # 값 종류가 너무 많으면(예: 숫자형 데이터) 개수만 표시
    if len(unique_vals) > 10:
        return f"연속형 숫자 ({len(unique_vals)}가지 값)"
    else:
        return ", ".join(map(str, sorted(unique_vals, key=str)))

column_info = pd.DataFrame({
    "열 이름": df.columns,
    "우리말 뜻": ["" for _ in df.columns],   # 학생이 직접 채워 넣을 빈 칸
    "값의 종류": [get_value_types(df[col]) for col in df.columns],
    "빈 값 개수": [df[col].isnull().sum() for col in df.columns]
})

# data_editor로 표시하면 '우리말 뜻' 칸을 직접 입력할 수 있음
edited_column_info = st.data_editor(
    column_info,
    use_container_width=True,
    num_rows="fixed",   # 행 추가/삭제는 못하게 고정
    disabled=["열 이름", "값의 종류", "빈 값 개수"],  # 우리말 뜻만 수정 가능
    hide_index=True
)

st.divider()

# ---------------------------------------------------
# 6. 데이터 처음 다섯 줄 보여주기
# ---------------------------------------------------
st.subheader("🔍 데이터 미리보기 (상위 5줄)")
st.dataframe(df.head(5), use_container_width=True)

st.divider()

# ---------------------------------------------------
# 7. 데이터 출처 적는 자리
#    - 학생이 교재를 보고 직접 입력할 수 있는 텍스트 입력창
# ---------------------------------------------------
st.subheader("📚 데이터 출처")
source_text = st.text_area(
    "교재를 참고하여 데이터 출처를 아래에 작성해보세요.",
    placeholder="여기에 데이터 출처를 입력하세요."
)

if source_text:
    st.info(source_text)
