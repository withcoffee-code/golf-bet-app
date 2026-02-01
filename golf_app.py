import streamlit as st
from itertools import combinations
from collections import Counter
import pandas as pd

# ----------------------
# 페이지 설정
# ----------------------
st.set_page_config(page_title="골프 내기 계산기 (완전판)", layout="centered")
st.title("⛳ 골프 내기 계산기 (버디/이글 자동 감지 + 배판 적용)")

# ----------------------
# 상태 저장
# ----------------------
if "players" not in st.session_state:
    st.session_state.players = ["A","B","C","D"]
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
# 플레이어 이름 입력
# ----------------------
st.subheader("👤 플레이어 이름 설정")
p1 = st.text_input("플레이어 1", st.session_state.players[0])
p2 = st.text_input("플레이어 2", st.session_state.players[1])
p3 = st.text_input("플레이어 3", st.session_state.players[2])
p4 = st.text_input("플레이어 4", st.session_state.players[3])

if st.button("이름 적용"):
    st.session_state.players = [p1,p2,p3,p4]
    st.success("플레이어 이름이 적용되었습니다!")

players = st.session_state.players

# ----------------------
# 룰 설정
# ----------------------
st.sidebar.header("⚙️ 룰 설정")
st.session_state.base_amount = st.sidebar.number_input(
    "기준금액 (타당)", min_value=1000, step=1000, value=st.session_state.base_amount
)
st.session_state.max_amount = st.sidebar.number_input(
    "홀당 최대 금액", min_value=5000, step=5000, value=st.session_state.max_amount
)

# ----------------------
# 현재 홀 점수 입력
# ----------------------
st.subheader(f"🏌️ 현재 홀: {st.session_state.hole} / 18")
par = st.selectbox("파", [3,4,5])
scores = [st.number_input(f"{p}",1,10,par) for p in players]

# ----------------------
# 1:1 + 배판 계산 함수
# ----------------------
def calculate_hole(scores, par, prev_all_tie, base_amount, max_amount):
    n = len(scores)
    # 버디/이글 자동 감지
    adj_scores = []
    for s in scores:
        diff = s - par
        # 버디/이글 보너스 적용
        if diff == -1:  # 버디
            diff -= 1
        elif diff <= -2:  # 이글
            diff -= 2
        adj_scores.append(diff)

    # 배판 결정
    counts = Counter(scores)
    tie_three = any(v >= 3 for v in counts.values())
    all_tie = len(set(scores)) == 1
    any_birdie_eagle = any((s - par) <= -1 for s in scores)
    batch_multiplier = 2 if tie_three or prev_all_tie or any_birdie_eagle else 1

    # 모든 플레이어 점수 같으면 금액 0
    if all_tie:
        money_matrix = [[0]*n for _ in range(n)]
        return [0]*n, money_matrix, all_tie

    # 1:1 금액 계산
    money_matrix = [[0]*n for _ in range(n)]
    for i,j in combinations(range(n),2):
        diff = adj_scores[j] - adj_scores[i]
        amt = diff * base_amount * batch_multiplier
        amt = max(-max_amount, min(max_amount, amt))
        money_matrix[i][j] = -amt
        money_matrix[j][i] = amt

    total_per_player = [sum(row) for row in money_matrix]
    return total_per_player, money_matrix, all_tie

# ----------------------
# 이번 홀 계산
# ----------------------
if st.button("이번 홀 계산"):
    totals, matrix, all_tie = calculate_hole(
        scores, par, st.session_state.prev_all_tie,
        st.session_state.base_amount, st.session_state.max_amount
    )

    # 누적 합산
    for i in range(4):
        st.session_state.total[i] += totals[i]

    # 기록
    st.session_state.history.append({
        "hole": st.session_state.hole,
        "scores": scores,
        "matrix": matrix,
        "totals": totals
    })

    st.session_state.prev_all_tie = all_tie
    st.session_state.hole += 1

    # 결과 출력
    st.subheader(f"홀 {st.session_state.hole-1} 결과")
    for i,p in enumerate(players):
        if totals[i] < 0:
            st.write(f"{p}: {abs(totals[i]):,}원 받음")
        else:
            st.write(f"{p}: {totals[i]:,}원 냄")

    # ----------------------
    # 1:1 시각화 매트릭스
    df = pd.DataFrame(matrix, index=players, columns=players)
    st.subheader("💰 1:1 금액 매트릭스 (이번 홀)")
    st.dataframe(df.style.format("{:,.0f}"))

# ----------------------
# 이전 홀 되돌리기
# ----------------------
if st.button("⬅ 이전 홀 되돌리기"):
    if st.session_state.history:
        last = st.session_state.history.pop()
        for i in range(4):
            st.session_state.total[i] -= last["totals"][i]
        st.session_state.hole -= 1

# ----------------------
# 최종 정산
# ----------------------
if st.session_state.hole > 18:
    st.subheader("🎉 라운드 종료! 최종 정산")
    for i,p in enumerate(players):
        if st.session_state.total[i] < 0:
            st.write(f"{p}: {abs(st.session_state.total[i]):,}원 받음")
        else:
            st.write(f"{p}: {st.session_state.total[i]:,}원 냄")
    if st.button("새 라운드 시작"):
        st.session_state.total = [0,0,0,0]
        st.session_state.hole = 1
        st.session_state.history = []
        st.session_state.prev_all_tie = False

# ----------------------
# 현재 누적
# ----------------------
st.divider()
st.subheader("📊 현재 누적")
for i,p in enumerate(players):
    if st.session_state.total[i] < 0:
        st.write(f"{p}: {abs(st.session_state.total[i]):,}원 받는 중")
    else:
        st.write(f"{p}: {st.session_state.total[i]:,}원 내는 중")
