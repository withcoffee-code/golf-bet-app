import streamlit as st
from itertools import combinations
from collections import Counter
import pandas as pd

# ----------------------
# 페이지 설정
# ----------------------
st.set_page_config(page_title="Kevin 룰 계산기", layout="centered")
st.title("⛳ Kevin 룰 계산기")

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
use_max_amount = st.sidebar.checkbox("홀당 최대 금액 적용", value=True)

# ----------------------
# 현재 홀 점수 입력 (드롭다운, 기본값 파)
# ----------------------
st.subheader(f"🏌️ 현재 홀: {st.session_state.hole} / 18")
par = st.selectbox("파", [3,4,5])

score_mapping = {
    "이글": -2,
    "버디": -1,
    "파": 0,
    "보기": 1,
    "더블": 2,
    "트리플": 3,
    "쿼드러플": 4
}

scores = []
score_labels = []
st.write("🏌️ 스코어 선택:")
for i, p in enumerate(players):
    sel = st.selectbox(f"{p} 스코어", list(score_mapping.keys()), index=2, key=f"score_{p}_{st.session_state.hole}")
    scores.append(par + score_mapping[sel])
    score_labels.append(sel)  # 결과 출력용 label 저장

# ----------------------
# 1:1 + 배판 계산 함수
# ----------------------
def calculate_hole(scores, par, prev_all_tie, base_amount, max_amount, use_max):
    n = len(scores)
    adj_scores = []
    multipliers = []
    reasons = []

    for s in scores:
        diff = s - par
        reason = []
        if diff == -1:
            diff -= 1
            multiplier = 2
            reason.append("버디 → 한타 추가, 배판")
        elif diff <= -2:
            diff -= 2
            multiplier = 4
            reason.append("이글 → 두타 추가, 배배판")
        else:
            multiplier = 1
            reason.append("일반")
        adj_scores.append(diff)
        multipliers.append(multiplier)
        reasons.append(", ".join(reason))

    counts = Counter(scores)
    tie_three = any(v >= 3 for v in counts.values())
    all_tie = len(set(scores)) == 1
    any_birdie_eagle = any((s - par) <= -1 for s in scores)
    batch_multiplier = 2 if tie_three or prev_all_tie or any_birdie_eagle else 1
    batch_reason = []
    if tie_three: batch_reason.append("3명 이상 동타 → 배판")
    if prev_all_tie: batch_reason.append("전홀 동타 → 배판")
    if any_birdie_eagle: batch_reason.append("이번 홀 버디/이글 → 배판")
    if not batch_reason: batch_reason.append("배판 없음")
    batch_reason_str = ", ".join(batch_reason)

    if all_tie:
        money_matrix = [[0]*n for _ in range(n)]
        return [0]*n, money_matrix, all_tie, reasons, batch_reason_str

    money_matrix = [[0]*n for _ in range(n)]
    for i,j in combinations(range(n),2):
        multiplier = max(multipliers[i], multipliers[j]) * batch_multiplier
        diff = adj_scores[j] - adj_scores[i]
        amt = diff * base_amount * multiplier
        if use_max:
            amt = max(-max_amount, min(max_amount, amt))
        money_matrix[i][j] = -amt
        money_matrix[j][i] = amt

    total_per_player = [sum(row) for row in money_matrix]
    return total_per_player, money_matrix, all_tie, reasons, batch_reason_str

# ----------------------
# 이번 홀 계산
# ----------------------
if st.button("이번 홀 계산"):
    totals, matrix, all_tie, reasons, batch_reason_str = calculate_hole(
        scores, par, st.session_state.prev_all_tie,
        st.session_state.base_amount, st.session_state.max_amount,
        use_max_amount
    )

    for i in range(4):
        st.session_state.total[i] += totals[i]

    st.session_state.history.append({
        "hole": st.session_state.hole,
        "scores": scores,
        "score_labels": score_labels,
        "matrix": matrix,
        "totals": totals
    })

    st.session_state.prev_all_tie = all_tie
    st.session_state.hole += 1

    st.subheader(f"홀 {st.session_state.hole-1} 결과")
    st.write(f"기본금액: {st.session_state.base_amount}원, 배판 설명: {batch_reason_str}")
    for i,p in enumerate(players):
        st.write(f"{p}: 스코어={score_labels[i]}, {reasons[i]}")
        if totals[i] < 0:
            st.write(f"→ {abs(totals[i]):,}원 받음")
        else:
            st.write(f"→ {totals[i]:,}원 냄")

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
