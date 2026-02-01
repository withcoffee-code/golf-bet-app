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
# 상태 초기화
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
if "max_per_stroke" not in st.session_state:
    st.session_state.max_per_stroke = 20000
if "apply_max_toggle" not in st.session_state:
    st.session_state.apply_max_toggle = True

# ----------------------
# 플레이어 이름 입력 (탭 선택 시 초기화)
# ----------------------
st.subheader("👤 플레이어 이름 설정")

def reset_player_input():
    for i in range(4):
        st.session_state[f"player_input_{i}"] = ""

p1 = st.text_input("플레이어 1", st.session_state.players[0], key="player_input_0", on_change=reset_player_input)
p2 = st.text_input("플레이어 2", st.session_state.players[1], key="player_input_1", on_change=reset_player_input)
p3 = st.text_input("플레이어 3", st.session_state.players[2], key="player_input_2", on_change=reset_player_input)
p4 = st.text_input("플레이어 4", st.session_state.players[3], key="player_input_3", on_change=reset_player_input)

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

st.session_state.apply_max_toggle = st.sidebar.checkbox(
    "타당 최대 금액 적용", value=st.session_state.apply_max_toggle
)

if st.session_state.apply_max_toggle:
    st.session_state.max_per_stroke = st.sidebar.number_input(
        "타당 최대 금액 (1타 기준)", min_value=1000, step=1000, value=st.session_state.max_per_stroke
)
else:
    st.session_state.max_per_stroke = None

# ----------------------
# 현재 홀 점수 입력
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
    score_labels.append(sel)

# ----------------------
# 홀 계산 함수
# ----------------------
def calculate_hole(scores, par, prev_all_tie, base_amount, max_per_stroke, score_labels):
    n = len(scores)

    # 1️⃣ 배판/배배판 적용 → 타당 금액 결정
    counts = Counter(scores)
    tie_three = any(v >= 3 for v in counts.values())
    all_tie = len(set(scores)) == 1
    any_birdie_eagle = any((s - par) <= -1 for s in scores)

    batch_multiplier = 1
    batch_reason = []

    if tie_three:
        batch_multiplier *= 2
        batch_reason.append("3명 이상 동타 → 배판 적용")
    if prev_all_tie:
        batch_multiplier *= 2
        batch_reason.append("전홀 동타 → 배판 적용")
    if any_birdie_eagle:
        batch_multiplier *= 2
        batch_reason.append("버디/이글 발생 → 배판 적용")
    if not batch_reason:
        batch_reason.append("배판 없음")

    batch_reason_str = "\n".join(batch_reason)

    # 2️⃣ 모든 스코어 동일 → 금액 없음
    if all_tie:
        money_matrix = [[0]*n for _ in range(n)]
        total_per_player = [0]*n
        return total_per_player, money_matrix, all_tie, batch_reason_str, batch_multiplier

    # 3️⃣ 1:1 금액 계산
    money_matrix = [[0]*n for _ in range(n)]
    for i,j in combinations(range(n),2):
        diff = scores[j] - scores[i]
        per_stroke_amount = base_amount * batch_multiplier
        if max_per_stroke:
            per_stroke_amount = min(per_stroke_amount, max_per_stroke)

        bonus = 0
        if score_labels[i] == "버디":
            bonus += 1
        elif score_labels[i] == "이글":
            bonus += 2
        if score_labels[j] == "버디":
            bonus -= 1
        elif score_labels[j] == "이글":
            bonus -= 2

        total_diff = diff + bonus
        amt = total_diff * per_stroke_amount

        money_matrix[i][j] = -amt
        money_matrix[j][i] = amt

    total_per_player = [sum(row) for row in money_matrix]
    return total_per_player, money_matrix, all_tie, batch_reason_str, batch_multiplier

# ----------------------
# 이번 홀 계산
# ----------------------
if st.button("이번 홀 계산"):
    totals, matrix, all_tie, batch_reason_str, batch_multiplier = calculate_hole(
        scores, par, st.session_state.prev_all_tie,
        st.session_state.base_amount, st.session_state.max_per_stroke,
        score_labels
    )

    for i in range(4):
        st.session_state.total[i] += totals[i]

    st.session_state.history.append({
        "hole": st.session_state.hole,
        "scores": scores,
        "score_labels": score_labels,
        "matrix": matrix,
        "totals": totals,
        "batch_multiplier": batch_multiplier
    })

    st.session_state.prev_all_tie = all_tie

    st.subheader(f"📝 홀 {st.session_state.hole} 처리 과정")
    st.markdown("**1️⃣ 타수 차 계산**")
    for i, s in enumerate(scores):
        diff = s - par
        st.write(f"{players[i]}: 스코어 {score_labels[i]} → 기본 타수 차 {diff:+}")

    st.markdown("**2️⃣ 버디/이글 보너스 적용 (1:1)**")
    for i, label in enumerate(score_labels):
        if label == "버디":
            st.write(f"{players[i]}: 버디 → 상대에게 1타 추가 금액")
        elif label == "이글":
            st.write(f"{players[i]}: 이글 → 상대에게 2타 추가 금액")
        else:
            st.write(f"{players[i]}: 보너스 없음")

    st.markdown("**3️⃣ 배판/배배판 적용**")
    st.write(batch_reason_str)
    st.write(f"▶ 적용 배수: {batch_multiplier}배")

    st.subheader("💰 이번 홀 최종 정리")
    hole_data = []
    for i,p in enumerate(players):
        status = "받음" if totals[i] < 0 else "냄" if totals[i] > 0 else "0원"
        amt = abs(totals[i])
        hole_data.append([p, score_labels[i], status, f"{amt:,}원"])
    df_hole = pd.DataFrame(hole_data, columns=["플레이어","스코어","상태","이번 홀 금액"])
    st.dataframe(df_hole)

    st.session_state.hole += 1

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
# 전체 리셋
# ----------------------
if st.button("🔄 전체 리셋"):
    st.session_state.total = [0,0,0,0]
    st.session_state.hole = 1
    st.session_state.history = []
    st.session_state.prev_all_tie = False
    st.success("전체 상태와 현재 홀이 초기화되었습니다!")

# ----------------------
# 현재 누적 총액 표시
# ----------------------
st.divider()
st.subheader("📊 현재 누적 총액")
for i, p in enumerate(players):
    amt = st.session_state.total[i]
    if amt < 0:
        st.write(f"{p}: {abs(amt):,}원 벌음")
    elif amt > 0:
        st.write(f"{p}: {amt:,}원 냄")
    else:
        st.write(f"{p}: 0원 (벌거나 냄 없음)")

# ----------------------
# 최종 정산 + 다음 라운드 핸디 계산 (사람별 합산)
# ----------------------
if st.session_state.hole > 18:
    st.subheader("🎉 라운드 종료! 최종 정산")
    for i,p in enumerate(players):
        amt = st.session_state.total[i]
        if amt < 0:
            st.write(f"{p}: {abs(amt):,}원 받음")
        elif amt > 0:
            st.write(f"{p}: {amt:,}원 냄")
        else:
            st.write(f"{p}: 0원 (벌거나 냄 없음)")

    st.subheader("📝 다음 라운드 핸디 총액 계산")
    n = len(players)
    total_scores = [sum(h["scores"][i] for h in st.session_state.history) for i in range(n)]
    hand_matrix = [[0]*n for _ in range(n)]
    for i,j in combinations(range(n),2):
        diff = total_scores[j] - total_scores[i]
        amt = diff * st.session_state.base_amount
        hand_matrix[i][j] = -amt
        hand_matrix[j][i] = amt

    hand_totals = [sum(row) for row in hand_matrix]

    hand_data = []
    for i,p in enumerate(players):
        amt = hand_totals[i]
        status = "받음" if amt < 0 else "냄" if amt > 0 else "0원"
        hand_data.append([p, total_scores[i], status, f"{abs(amt):,}원"])

    df_hand = pd.DataFrame(hand_data, columns=["플레이어","총 타수","상태","핸디 총액"])
    st.write(f"기본 타당 금액: {st.session_state.base_amount}원")
    st.dataframe(df_hand)

    if st.button("새 라운드 시작"):
        st.session_state.total = [0,0,0,0]
        st.session_state.hole = 1
        st.session_state.history = []
        st.session_state.prev_all_tie = False
