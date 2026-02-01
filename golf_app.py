import streamlit as st
from collections import Counter
import openai
import json

# ----------------------
# 페이지 설정
# ----------------------
st.set_page_config(page_title="골프 내기 계산기", layout="centered")
st.title("⛳ 골프 내기 계산기 (완전판)")

# ----------------------
# 상태 저장
# ----------------------
if "total" not in st.session_state:
    st.session_state.total = [0,0,0,0]
if "prev_all_tie" not in st.session_state:
    st.session_state.prev_all_tie = False
if "hole" not in st.session_state:
    st.session_state.hole = 1
if "history" not in st.session_state:
    st.session_state.history = []
if "base_amount" not in st.session_state:
    st.session_state.base_amount = 5000
if "max_amount" not in st.session_state:
    st.session_state.max_amount = 20000

# ----------------------
# 사이드바 - 룰 설정
# ----------------------
st.sidebar.header("⚙️ 룰 설정")
st.session_state.base_amount = st.sidebar.number_input(
    "기준금액 (타당)",
    min_value=1000, step=1000, value=st.session_state.base_amount
)
st.session_state.max_amount = st.sidebar.number_input(
    "홀당 최대 금액",
    min_value=5000, step=5000, value=st.session_state.max_amount
)
use_birdie_bonus = st.sidebar.checkbox("버디 보너스 적용", value=True)
use_eagle_bonus = st.sidebar.checkbox("이글 보너스 적용", value=True)
st.sidebar.markdown("---")
st.sidebar.write("현재 룰 요약")
st.sidebar.write(f"- 기준금액: {st.session_state.base_amount:,}원")
st.sidebar.write(f"- 최대금액: {st.session_state.max_amount:,}원")
st.sidebar.write(f"- 버디보너스: {'ON' if use_birdie_bonus else 'OFF'}")
st.sidebar.write(f"- 이글보너스: {'ON' if use_eagle_bonus else 'OFF'}")

# ----------------------
# OpenAI API Key
# ----------------------
openai.api_key = st.text_input("OpenAI API Key", type="password")

# ----------------------
# AI 점수 입력
# ----------------------
st.subheader("🗣 점수 말로 입력")
text_input = st.text_area("예: 파4 A버디 B파 C파 D보기")

def parse_with_ai(text):
    prompt = f"""
너는 골프 점수 파서다.
아래 문장을 JSON으로 바꿔라.
형식: {{"par":4,"scores":[3,4,4,5]}}
문장: {text}
"""
    response = openai.ChatCompletion.create(
        model="gpt-4.1-mini",
        messages=[{"role":"user","content":prompt}]
    )
    return json.loads(response.choices[0].message.content)

# ----------------------
# 수동 입력
# ----------------------
st.subheader("⌨ 수동 입력")
par = st.selectbox("파", [3,4,5])
scores = [
    st.number_input("A",1,10,4),
    st.number_input("B",1,10,4),
    st.number_input("C",1,10,4),
    st.number_input("D",1,10,4)
]

# ----------------------
# 계산 함수
# ----------------------
def calculate_hole(par, scores, prev_all_tie, base_amount, max_amount):
    diff = [s - par for s in scores]
    birdie_count = 0
    final_diff = []
    for d in diff:
        if d == -1 and use_birdie_bonus:
            final_diff.append(-2)
            birdie_count += 1
        elif d <= -2 and use_eagle_bonus:
            final_diff.append(-4)
            birdie_count += 2
        else:
            final_diff.append(d)
    counts = Counter(scores)
    tie_three = any(v >= 3 for v in counts.values())
    double_count = birdie_count
    if tie_three: double_count += 1
    if prev_all_tie: double_count += 1
    multiplier = min(2 ** double_count, 4)
    unit_money = min(base_amount * multiplier, max_amount)
    result = [d * unit_money for d in final_diff]
    all_tie = len(set(scores)) == 1
    return unit_money, result, all_tie

# ----------------------
# 홀 계산
# ----------------------
st.subheader(f"🏌️ 현재 홀: {st.session_state.hole} / 18")

if st.button("이번 홀 계산"):
    if text_input and openai.api_key:
        data = parse_with_ai(text_input)
        par = data["par"]
        scores = data["scores"]
    unit_money, result, all_tie = calculate_hole(
        par,
        scores,
        st.session_state.prev_all_tie,
        st.session_state.base_amount,
        st.session_state.max_amount
    )
    st.session_state.history.append({
        "hole": st.session_state.hole,
        "unit": unit_money,
        "result": result
    })
    for i in range(4):
        st.session_state.total[i] += result[i]
    st.session_state.prev_all_tie = all_tie
    st.session_state.hole += 1

    # 이번 홀 결과 출력
    players = ["A","B","C","D"]
    st.subheader(f"이번 홀 결과 (타당: {unit_money:,}원)")
    for i,p in enumerate(players):
        if result[i] < 0:
            st.write(f"{p}: {abs(result[i]):,}원 받음")
        else:
            st.write(f"{p}: {result[i]:,}원 냄")

# ----------------------
# 이전 홀 되돌리기
# ----------------------
if st.button("⬅ 이전 홀 되돌리기"):
    if st.session_state.history:
        last = st.session_state.history.pop()
        for i in range(4):
            st.session_state.total[i] -= last["result"][i]
        st.session_state.hole -= 1

# ----------------------
# 18홀 종료 시 최종 정산
# ----------------------
if st.session_state.hole > 18:
    st.subheader("🎉 라운드 종료! 최종 정산")
    players = ["A","B","C","D"]
    for p,t in zip(players, st.session_state.total):
        if t < 0:
            st.write(f"{p}: {abs(t):,}원 받음")
        else:
            st.write(f"{p}: {t:,}원 냄")
    if st.button("새 라운드 시작"):
        st.session_state.total = [0,0,0,0]
        st.session_state.prev_all_tie = False
        st.session_state.hole = 1
        st.session_state.history = []
        st.experimental_rerun()

# ----------------------
# 누적 정산
# ----------------------
st.divider()
st.subheader("📊 현재 누적")
players = ["A","B","C","D"]
for p,t in zip(players, st.session_state.total):
    if t < 0:
        st.write(f"{p}: {abs(t):,}원 받는 중")
    else:
        st.write(f"{p}: {t:,}원 내는 중")
