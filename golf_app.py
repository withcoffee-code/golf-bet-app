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
st.session_state.max_per_stroke = st.sidebar.number_input(
    "타당 최대 금액 (1타 기준)", min_value=1000, step=1000, value=st.session_state.max_per_stroke
)

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
# 1:1 + 배판 계산 함수
# ----------------------
def calculate_hole(scores, par, prev_all_tie, base_amount, max_per_stroke):
    n = len(scores)
    multipliers = []
    reasons = []

    for s in scores:
        diff = s - par
        multiplier = 1
        reason = []
        if diff == -1:  # 버디
            multiplier = 2
            reason.append("버디 → 한타 추가")
        elif diff <= -2:  # 이글
            multiplier = 4
            reason.append("이글 → 두타 추가")
        else:
            reason.append("일반")
        multipliers.append(multiplier)
        reasons.append(", ".join(reason))

    # 배판 판단
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
    batch_reason_str = "\n".join(batch_reason)

    if all_tie:
        money_matrix = [[0]*n for _ in range(n)]
        total_per_player = [0]*n
        return total_per_player, money_matrix, all_tie, reasons, batch_reason_str

    # 금액 매트릭스 계산 (타당 최대금액 적용)
    money_matrix = [[0]*n for _ in range(n)]
    for i,j in combinations(range(n),2):
        diff = scores[j] - scores[i]  # 실제 타수 차이
        per_stroke_amount = min(base_amount * max(multipliers[i], multipliers[j]), max_per_stroke)
        amt = diff * per_stroke_amount * batch_multiplier
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
        st.session_state.base_amount, st.session_state.max_per_stroke
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

    # 이번 홀 결과 그리드
    hole_data = []
    for i,p in enumerate(players):
        status = "받음" if totals[i] < 0 else "냄" if totals[i] > 0 else "0원"
        amt = abs(totals[i])
        hole_data.append([p, score_labels[i], status, f"{amt:,}원"])

    df_hole = pd.DataFrame(hole_data, columns=["플레이어","스코어","상태","이번 홀 금액"])
    st.subheader(f"🏌️ 홀 {st.session_state.hole} 결과")

    # 배판 + 보너스 설명
    bonus_text = []
    for i,r in enumerate(reasons):
        bonus_text.append(f"{players[i]}: {r}")
    description = f"**기본금액:** {st.session_state.base_amount:,}원  \n"
    description += f"**배판 설명:**  \n{batch_reason_str}  \n"
    description += "**버디/이글 보너스:**  \n" + "\n".join(bonus_text)
    st.markdown(description.replace("\n","  \n"))

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
# 최종 정산 + 다음 라운드 핸디 계산
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

    # 다음 라운드 핸디 계산
    st.subheader("📝 다음 라운드 핸디 금액 계산")
    n = len(players)
    total_scores = [sum(h["scores"][i] for h in st.session_state.history) for i in range(n)]
    hand_matrix = [[0]*n for _ in range(n)]
    for i,j in combinations(range(n),2):
        diff = total_scores[j] - total_scores[i]
        amt = diff * st.session_state.base_amount
        hand_matrix[i][j] = -amt
        hand_matrix[j][i] = amt

    df_hand = pd.DataFrame(hand_matrix, index=players, columns=players)
    st.write(f"기본 타당 금액: {st.session_state.base_amount}원")
    st.dataframe(df_hand.style.format("{:,.0f}"))

    if st.button("새 라운드 시작"):
        st.session_state.total = [0,0,0,0]
        st.session_state.hole = 1
        st.session_state.history = []
        st.session_state.prev_all_tie = False
