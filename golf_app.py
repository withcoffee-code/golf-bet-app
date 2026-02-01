import streamlit as st
from collections import Counter

# ----------------------
# 페이지 설정
# ----------------------
st.set_page_config(page_title="골프 내기 계산기", layout="centered")
st.title("⛳ 골프 내기 계산기 (완전판, AI 없음)")

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
    "기준금액 (타당)", min_value=1000, step=1000, value=st.session_state.base_amount
)
st.session_state.max_amount = st.sidebar.number_input(
    "홀당 최대 금액", min_value=5000, step=5000, value=st.session_state.max_amount
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
# 수동 입력
# ----------------------
st.subheader(f"🏌️ 현재 홀: {st.session_state.hole} / 18")
par = st.selectbox("파", [3,4,5])
scores = [
    st.number_input("A",1,10,par),
    st.number_input("B",1,10,par),
    st.number_input("C",1,10,par),
    st.number_input("D",1,10,par)
]

# ----------------------
# 계산 함수
# ----------------------
def calculate_hole_fixed(par, scores, prev_all_tie, base_amount, max_amount):
    counts = Counter(scores)
    tie_three = any(v >= 3 for v in counts.values())
    all_tie = len(set(scores)) == 1

    results = []
    for s in scores:
        d = s - par
        personal_double = 0
        if d == -1 and use_birdie_bonus:    # 버디 보너스
            personal_double += 1
        elif d <= -2 and use_eagle_bonus:   # 이글 보너스
            personal_double += 2

        # 동타/전홀 배판
        if tie_three:
            personal_double += 1
        if prev_all_tie:
            personal_double += 1

        multiplier = min(2 ** personal_double, 4)
        unit_money = min(base_amount * multiplier, max_amount)

        results.append(d * unit_money)

    return results, all_tie

# ----------------------
# 홀 계산
# ----------------------
if st.button("이번 홀 계산"):
    results, all_tie = calculate_hole_fixed(
        par, scores, st.session_state.prev_all_tie,
        st.session_state.base_amount, st.session_state.max_amount
    )

    st.session_state.history.append({
        "hole": st.session_state.hole,
        "result": results
    })

    # 누적 반영
    for i in range(4):
        st.session_state.total[i] += results[i]

    st.session_state.prev_all_tie = all_tie
    st.session_state.hole += 1

    # 이번 홀 결과 출력
    players = ["A","B","C","D"]
    st.subheader("이번 홀 결과")
    for i,p in enumerate(players):
        if results[i] < 0:
            st.write(f"{p}: {abs(results[i]):,}원 받음")
        else:
            st.write(f"{p}: {results[i]:,}원 냄")

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
