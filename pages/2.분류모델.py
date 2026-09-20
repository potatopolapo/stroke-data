import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

# ---------------------------------------------------
# 1. 페이지 기본 설정
# ---------------------------------------------------
st.set_page_config(
    page_title="뇌졸중 예측 실습실 - 분류 모델",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 분류 모델 만들기")
st.write("로지스틱 회귀와 의사결정트리 모델을 만들어 뇌졸중을 예측해봅니다.")

st.divider()

# ---------------------------------------------------
# 2. 데이터 불러오기
# ---------------------------------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
    df = pd.read_csv(url, encoding="utf-8")
    return df

df = load_data()

# ---------------------------------------------------
# 3. 열 이름 <-> 우리말 이름 매핑
# ---------------------------------------------------
col_to_kor = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병"
}
kor_to_col = {v: k for k, v in col_to_kor.items()}

all_features = ["age", "avg_glucose_level", "bmi", "hypertension", "heart_disease"]
default_features = ["age", "avg_glucose_level", "hypertension", "heart_disease"]  # bmi 제외

# ---------------------------------------------------
# 4. 속성 선택 위젯 (우리말 이름으로 표시)
# ---------------------------------------------------
st.subheader("1️⃣ 입력 속성 선택")

selected_kor = st.multiselect(
    "모델의 입력으로 사용할 속성을 고르세요.",
    options=[col_to_kor[c] for c in all_features],
    default=[col_to_kor[c] for c in default_features]
)

selected_features = [kor_to_col[k] for k in selected_kor]

if len(selected_features) < 2:
    st.warning("⚠️ 속성을 두 개 이상 선택해야 합니다. 목록에서 속성을 더 골라주세요.")
    st.stop()

st.divider()

# ---------------------------------------------------
# 5. 데이터 준비: 번호 순 정렬 -> 10명씩 묶어 앞 3명 테스트용
# ---------------------------------------------------
st.subheader("2️⃣ 학습용·테스트용 데이터 나누기")

df_sorted = df.sort_values("id").reset_index(drop=True)

# 10명씩 묶었을 때 그룹 안에서의 순서 (0~9)
position_in_group = np.arange(len(df_sorted)) % 10

# 각 묶음의 앞 3명(위치 0,1,2)은 테스트용, 나머지 7명은 학습용
is_test = position_in_group < 3

test_df = df_sorted[is_test].copy()
train_df = df_sorted[~is_test].copy()

st.write(f"- 전체 사람 수: **{len(df_sorted):,} 명**")
st.write(f"- 학습용 사람 수: **{len(train_df):,} 명**")
st.write(f"- 테스트용 사람 수: **{len(test_df):,} 명**")

# ---------------------------------------------------
# 6. bmi 결측치 처리 (bmi를 선택한 경우에만, 훈련용 중앙값 사용)
# ---------------------------------------------------
if "bmi" in selected_features:
    bmi_median = train_df["bmi"].median()
    train_df["bmi"] = train_df["bmi"].fillna(bmi_median)
    test_df["bmi"] = test_df["bmi"].fillna(bmi_median)
    st.write(f"- 체질량지수(bmi) 빈 값은 훈련용 중앙값 **{bmi_median:.2f}** 로 채웠습니다.")

X_train = train_df[selected_features]
y_train = train_df["stroke"]
X_test = test_df[selected_features]
y_test = test_df["stroke"]

st.divider()

# ---------------------------------------------------
# 7. 모델 학습
#    - 로지스틱 회귀
#    - 의사결정트리 (질문 3번까지, 마지막 마디 5명 미만이면 그만)
#    - 더미(기준) 모델: 많은 쪽으로만 답하는 모델
# ---------------------------------------------------
log_model = LogisticRegression(random_state=42, max_iter=1000)
log_model.fit(X_train, y_train)

tree_model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=42
)
tree_model.fit(X_train, y_train)

dummy_model = DummyClassifier(strategy="most_frequent", random_state=42)
dummy_model.fit(X_train, y_train)

# 정확도 계산 함수
def get_accuracies(model, X_train, y_train, X_test, y_test):
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    return train_acc, test_acc

log_train_acc, log_test_acc = get_accuracies(log_model, X_train, y_train, X_test, y_test)
tree_train_acc, tree_test_acc = get_accuracies(tree_model, X_train, y_train, X_test, y_test)
dummy_train_acc, dummy_test_acc = get_accuracies(dummy_model, X_train, y_train, X_test, y_test)

# ---------------------------------------------------
# 8. 정확도 카드 세 개
# ---------------------------------------------------
st.subheader("3️⃣ 모델별 정확도 비교")

card1, card2, card3 = st.columns(3)

with card1:
    st.metric(label="로지스틱 회귀 (확률로 답하는 모델)", value=f"{log_test_acc*100:.2f} %")
    st.caption(f"훈련 정확도: {log_train_acc*100:.2f} %  ㅣ  테스트 정확도: {log_test_acc*100:.2f} %")

with card2:
    st.metric(label="의사결정트리 (질문으로 답하는 모델)", value=f"{tree_test_acc*100:.2f} %")
    st.caption(f"훈련 정확도: {tree_train_acc*100:.2f} %  ㅣ  테스트 정확도: {tree_test_acc*100:.2f} %")

with card3:
    st.metric(label="기준 모델 (많은 쪽으로만 답하는 모델)", value=f"{dummy_test_acc*100:.2f} %")
    st.caption(f"훈련 정확도: {dummy_train_acc*100:.2f} %  ㅣ  테스트 정확도: {dummy_test_acc*100:.2f} %")

st.divider()

# ---------------------------------------------------
# 9. 산점도: 두 속성을 축으로, 로지스틱 회귀 경계선 표시
# ---------------------------------------------------
st.subheader("4️⃣ 산점도와 로지스틱 회귀 경계선")

axis_col1, axis_col2 = st.columns(2)
with axis_col1:
    x_axis_kor = st.selectbox("가로축(X축)으로 사용할 속성", options=selected_kor, index=0)
with axis_col2:
    remaining_kor = [k for k in selected_kor if k != x_axis_kor]
    y_axis_kor = st.selectbox("세로축(Y축)으로 사용할 속성", options=remaining_kor, index=0)

x_axis = kor_to_col[x_axis_kor]
y_axis = kor_to_col[y_axis_kor]

# 두 축이 아닌 나머지 속성은 테스트 데이터의 중앙값으로 고정
other_features = [f for f in selected_features if f not in [x_axis, y_axis]]
fixed_values = {f: X_test[f].median() for f in other_features}

if fixed_values:
    fixed_text = ", ".join([f"{col_to_kor[k]} = {v:.2f}" for k, v in fixed_values.items()])
    st.write(f"📌 축으로 사용하지 않은 속성은 테스트 데이터의 중앙값으로 고정했습니다: **{fixed_text}**")
else:
    st.write("📌 선택한 속성이 두 개뿐이라 고정할 속성이 없습니다.")

# 테스트 데이터 산점도 그리기
scatter_df = test_df.copy()
scatter_df["뇌졸중 여부"] = scatter_df["stroke"].map({0: "뇌졸중 없음", 1: "뇌졸중 있음"})

fig_scatter = px.scatter(
    scatter_df, x=x_axis, y=y_axis,
    color="뇌졸중 여부",
    labels={x_axis: x_axis_kor, y_axis: y_axis_kor},
    title="테스트 데이터 산점도 (실제 뇌졸중 여부)",
    opacity=0.7
) if False else None  # placeholder, 실제 사용은 go로 통일

import plotly.express as px

fig_scatter = px.scatter(
    scatter_df, x=x_axis, y=y_axis,
    color="뇌졸중 여부",
    labels={x_axis: x_axis_kor, y_axis: y_axis_kor},
    title="테스트 데이터 산점도 (실제 뇌졸중 여부)",
    opacity=0.7
)

# --- 로지스틱 회귀 0.5 결정 경계선 계산 ---
# 로지스틱 회귀 식: w1*x1 + w2*x2 + ... + b = 0 일 때 확률 0.5
# 여기서는 x_axis, y_axis를 제외한 나머지는 고정값으로 넣고
# x_axis에 따른 y_axis 값을 계산해서 직선을 그림
coef = log_model.coef_[0]
intercept = log_model.intercept_[0]
feature_order = list(X_train.columns)

x_idx = feature_order.index(x_axis)
y_idx = feature_order.index(y_axis)

# 고정된 값들의 기여도(상수항에 합쳐줌)
constant = intercept
for f in other_features:
    f_idx = feature_order.index(f)
    constant += coef[f_idx] * fixed_values[f]

# w_x * x + w_y * y + constant = 0  ->  y = -(w_x * x + constant) / w_y
w_x = coef[x_idx]
w_y = coef[y_idx]

x_min, x_max = X_test[x_axis].min(), X_test[x_axis].max()
x_range = np.linspace(x_min, x_max, 100)

line_out_of_range = False

if abs(w_y) < 1e-10:
    # y 계수가 거의 0이면 y에 대해 풀 수 없음 -> 세로선 형태
    if abs(w_x) < 1e-10:
        st.write("📌 선택한 두 속성의 계수가 모두 0에 가까워 경계선을 그릴 수 없습니다.")
        line_out_of_range = True
    else:
        x_boundary = -constant / w_x
        y_min, y_max = X_test[y_axis].min(), X_test[y_axis].max()
        if x_boundary < x_min or x_boundary > x_max:
            line_out_of_range = True
        fig_scatter.add_trace(go.Scatter(
            x=[x_boundary, x_boundary], y=[y_min, y_max],
            mode="lines", name="결정 경계(0.5)",
            line=dict(color="black", dash="dash")
        ))
else:
    y_range = -(w_x * x_range + constant) / w_y
    y_min, y_max = X_test[y_axis].min(), X_test[y_axis].max()

    # 경계선이 그림 범위를 벗어나는지 확인
    if y_range.min() > y_max or y_range.max() < y_min:
        line_out_of_range = True
    else:
        fig_scatter.add_trace(go.Scatter(
            x=x_range, y=y_range,
            mode="lines", name="결정 경계(0.5)",
            line=dict(color="black", dash="dash")
        ))

if line_out_of_range:
    st.write("📌 로지스틱 회귀의 결정 경계선이 그림 범위를 벗어나 화면에 표시되지 않습니다.")

# --- 의사결정트리의 영역을 배경색으로 칠하기 ---
x_grid_min, x_grid_max = X_test[x_axis].min(), X_test[x_axis].max()
y_grid_min, y_grid_max = X_test[y_axis].min(), X_test[y_axis].max()

grid_size = 100
xx, yy = np.meshgrid(
    np.linspace(x_grid_min, x_grid_max, grid_size),
    np.linspace(y_grid_min, y_grid_max, grid_size)
)

# 격자용 데이터프레임 만들기 (다른 속성은 고정값 사용)
grid_df = pd.DataFrame({x_axis: xx.ravel(), y_axis: yy.ravel()})
for f in other_features:
    grid_df[f] = fixed_values[f]
grid_df = grid_df[feature_order]  # 학습 시 열 순서 맞추기

grid_pred = tree_model.predict(grid_df)
grid_pred = grid_pred.reshape(xx.shape)

fig_scatter.add_trace(go.Contour(
    x=np.linspace(x_grid_min, x_grid_max, grid_size),
    y=np.linspace(y_grid_min, y_grid_max, grid_size),
    z=grid_pred,
    showscale=False,
    opacity=0.25,
    colorscale=[[0, "blue"], [1, "red"]],
    contours=dict(coloring="fill"),
    name="의사결정트리 영역",
    hoverinfo="skip"
))

# 배경(Contour)이 점 위에 그려지지 않도록 순서 조정
fig_scatter.data = fig_scatter.data[::-1]

st.plotly_chart(fig_scatter, use_container_width=True)

st.divider()

# ---------------------------------------------------
# 10. 의사결정트리 가지 그림 (graphviz DOT 문자열)
# ---------------------------------------------------
st.subheader("5️⃣ 의사결정트리 가지 그림")

tree = tree_model.tree_
feature_names = list(X_train.columns)

def build_dot(tree, feature_names, kor_map):
    dot_lines = ["digraph Tree {", 'node [shape=box, style="filled", fontname="Malgun Gothic"];']

    n_nodes = tree.node_count
    children_left = tree.children_left
    children_right = tree.children_right
    feature = tree.feature
    threshold = tree.threshold
    value = tree.value  # [노드][클래스별 개수]

    for i in range(n_nodes):
        n_samples = int(value[i].sum())
        n_positive = int(value[i][0][1])  # stroke=1인 개수
        ratio = n_positive / n_samples if n_samples > 0 else 0

        is_leaf = children_left[i] == children_right[i]  # 둘 다 -1이면 leaf

        if is_leaf:
            # 리프 노드: 다수결로 최종 답 결정
            predicted_class = 1 if value[i][0][1] > value[i][0][0] else 0
            color = "#ffcccc" if predicted_class == 1 else "#cce5ff"
            label = f"인원 {n_samples}명\\n뇌졸중 {n_positive}명\\n비율 {ratio*100:.1f}%\\n답: {'뇌졸중' if predicted_class==1 else '아님'}"
            dot_lines.append(f'{i} [label="{label}", fillcolor="{color}"];')
        else:
            feat_name = kor_map[feature_names[feature[i]]]
            thresh = threshold[i]
            label = f"{feat_name} <= {thresh:.2f} ?\\n인원 {n_samples}명\\n뇌졸중 {n_positive}명\\n비율 {ratio*100:.1f}%"
            dot_lines.append(f'{i} [label="{label}", fillcolor="#ffffcc"];')

    # 가지(엣지) 그리기
    for i in range(n_nodes):
        if children_left[i] != children_right[i]:  # leaf가 아니면
            dot_lines.append(f'{i} -> {children_left[i]} [label="예"];')
            dot_lines.append(f'{i} -> {children_right[i]} [label="아니요"];')

    dot_lines.append("}")
    return "\n".join(dot_lines)

dot_string = build_dot(tree, feature_names, col_to_kor)
st.graphviz_chart(dot_string)

st.divider()

# ---------------------------------------------------
# 11. 트리 요약 정보
# ---------------------------------------------------
st.subheader("6️⃣ 의사결정트리 요약")

children_left = tree.children_left
children_right = tree.children_right
value = tree.value
feature = tree.feature

leaf_indices = [i for i in range(tree.node_count) if children_left[i] == children_right[i]]
n_leaves = len(leaf_indices)

n_negative_leaves = 0
for i in leaf_indices:
    predicted_class = 1 if value[i][0][1] > value[i][0][0] else 0
    if predicted_class == 0:
        n_negative_leaves += 1

used_feature_indices = set(feature[i] for i in range(tree.node_count) if children_left[i] != children_right[i])
used_features_kor = [col_to_kor[feature_names[idx]] for idx in used_feature_indices]

st.write(f"- 답을 내는 마디(리프 노드)는 모두 **{n_leaves} 칸**이고, 그중 **{n_negative_leaves} 칸**이 '아님'이라고 답합니다.")

if used_features_kor:
    st.write(f"- 고른 속성 가운데 이 나무가 실제로 물어본 것: **{', '.join(used_features_kor)}**")
else:
    st.write("- 이 나무는 어떤 속성도 질문에 사용하지 않았습니다. (모든 사람이 같은 답을 받는 경우)")
