# app.py
import os
import base64
import io
import json
from collections import Counter
from itertools import combinations

import pandas as pd
import streamlit as st
from PIL import Image, ImageEnhance, ImageOps

from openai import OpenAI


# ======================
# Page
# ======================
st.set_page_config(page_title="Kevin 룰 계산기", layout="wide")
st.title("⛳ Kevin 룰 계산기 (합계검증 + 홀별 정산)")

MAX_WIDTH = 1800


# ======================
# API
# ======================
def get_api_key():
    try:
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY")


def get_client():
    key = get_api_key()
    if not key:
        st.error("❌ OPENAI_API_KEY가 설정되지 않았습니다. (Secrets 또는 환경변수)")
        st.stop()
    return OpenAI(api_key=key)


# ======================
# Image utils (✅ 크롭/2분할 기능 제거)
# ======================
def _resize_cap(img: Image.Image, max_w=MAX_WIDTH) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    if w > max_w:
        ratio = max_w / w
        img = img.resize((int(w * ratio), int(h * ratio)), Image.Resampling.LANCZOS)
    return img


def preprocess_light(img: Image.Image) -> Image.Image:
    img = _resize_cap(img, MAX_WIDTH)
    gray = ImageOps.grayscale(img)
    gray = ImageOps.autocontrast(gray)
    gray = ImageEnhance.Contrast(gray).enhance(1.25)
    return gray.convert("RGB")


def preprocess_strong(img: Image.Image) -> Image.Image:
    img = _resize_cap(img, MAX_WIDTH)
    gray = ImageOps.grayscale(img)
    gray = ImageOps.autocontrast(gray)
    gray = ImageEnhance.Contrast(gray).enhance(1.6)
    gray = ImageEnhance.Sharpness(gray).enhance(1.4)
    gray = gray.point(lambda p: 255 if p > 200 else 0)
    return gray.convert("RGB")


def to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


# ======================
# JSON Schemas (strict)
# ======================
STRUCT_SCHEMA = {
    "name": "scorecard_structure",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "players": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 4,
                "maxItems": 4,
            },
            "out_pars": {
                "type": "array",
                "items": {"type": "integer", "enum": [3, 4, 5]},
                "minItems": 9,
                "maxItems": 9,
            },
            "in_pars": {
                "type": "array",
                "items": {"type": "integer", "enum": [3, 4, 5]},
                "minItems": 9,
                "maxItems": 9,
            },
        },
        "required": ["players", "out_pars", "in_pars"],
    },
    "strict": True,
}

STROKES_SCHEMA = {
    "name": "scorecard_strokes",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "out_strokes": {
                "type": "array",
                "minItems": 4,
                "maxItems": 4,
                "items": {
                    "type": "array",
                    "minItems": 9,
                    "maxItems": 9,
                    "items": {"type": "integer", "minimum": -1, "maximum": 20},
                },
            },
            "in_strokes": {
                "type": "array",
                "minItems": 4,
                "maxItems": 4,
                "items": {
                    "type": "array",
                    "minItems": 9,
                    "maxItems": 9,
                    "items": {"type": "integer", "minimum": -1, "maximum": 20},
                },
            },
        },
        "required": ["out_strokes", "in_strokes"],
    },
    "strict": True,
}

TOTALS_SCHEMA = {
    "name": "scorecard_totals",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "out_total": {"type": "array", "minItems": 4, "maxItems": 4, "items": {"type": "integer", "minimum": -1, "maximum": 200}},
            "in_total": {"type": "array", "minItems": 4, "maxItems": 4, "items": {"type": "integer", "minimum": -1, "maximum": 200}},
            "grand_total": {"type": "array", "minItems": 4, "maxItems": 4, "items": {"type": "integer", "minimum": -1, "maximum": 200}},
        },
        "required": ["out_total", "in_total", "grand_total"],
    },
    "strict": True,
}


# ======================
# OpenAI call helper (fallback)
# ======================
def call_json_schema(content, schema_pack, model_primary="gpt-4.1", model_fallback="gpt-4.1-mini"):
    client = get_client()
    try:
        resp = client.responses.create(
            model=model_primary,
            input=[{"role": "user", "content": content}],
            text={"format": {"type": "json_schema", "name": schema_pack["name"], "schema": schema_pack["schema"], "strict": True}},
        )
        return json.loads(resp.output_text)
    except Exception:
        resp = client.responses.create(
            model=model_fallback,
            input=[{"role": "user", "content": content}],
            text={"format": {"type": "json_schema", "name": schema_pack["name"], "schema": schema_pack["schema"], "strict": True}},
        )
        return json.loads(resp.output_text)


# ======================
# Extraction (2-step: structure -> strokes)
# ======================
def extract_structure(img: Image.Image):
    prompt = """
이 이미지는 골프 스코어카드(한 장에 OUT/IN 모두 포함)이다.

반드시 아래만 추출해라:
- players: 4명 이름(스코어 표에서 타수가 기록된 순서대로)
- out_pars: OUT 1~9홀 PAR 9개
- in_pars: IN 10~18홀 PAR 9개

규칙:
- OUT/IN 섞지 마라.
- PAR는 3/4/5만.
- JSON 스키마 외 출력 금지.
"""
    content = [{"type": "input_text", "text": prompt}, {"type": "input_image", "image_url": to_data_url(img)}]
    return call_json_schema(content, STRUCT_SCHEMA)


def extract_strokes(img: Image.Image, players, out_pars, in_pars):
    prompt = f"""
이 이미지는 골프 스코어카드(한 장에 OUT/IN 모두 포함)이다.
이미 아래 정보는 확정되었다. 절대 변경하지 마라.

players(순서 고정): {players}
out_pars(1~9): {out_pars}
in_pars(10~18): {in_pars}

너는 이제 '타수 숫자만' 읽어서 반환하면 된다.

반드시 아래만 출력:
- out_strokes: players 순서대로, 각 플레이어의 OUT 타수 9개 (총 4x9)
- in_strokes: players 순서대로, 각 플레이어의 IN 타수 9개 (총 4x9)

규칙:
- 숫자를 확실히 읽을 수 없으면 추정하지 말고 -1로 넣어라.
- 동그라미/별/아이콘은 숫자가 아니다. 무시하라.
- OUT/IN 섞지 마라.
- JSON 스키마 외 출력 금지.
"""
    content = [{"type": "input_text", "text": prompt}, {"type": "input_image", "image_url": to_data_url(img)}]
    return call_json_schema(content, STROKES_SCHEMA)


def extract_totals(img: Image.Image, players):
    prompt = f"""
이 이미지는 골프 스코어카드이다.
players 순서는 고정이다.

players: {players}

카드에 OUT/IN/TOTAL 합계 칸이 있으면 읽어라:
- out_total / in_total / grand_total (각 4명)
없거나 확실히 못 읽으면 -1.
추정 금지. JSON 외 출력 금지.
"""
    content = [{"type": "input_text", "text": prompt}, {"type": "input_image", "image_url": to_data_url(img)}]
    return call_json_schema(content, TOTALS_SCHEMA)


# ======================
# DF build + checks
# ======================
def to_df18(players, out_pars, in_pars, out_strokes, in_strokes):
    rows = []
    for idx in range(9):
        row = {"Hole": idx + 1, "Par": int(out_pars[idx])}
        for p_i, name in enumerate(players):
            row[name] = int(out_strokes[p_i][idx])
        rows.append(row)
    for idx in range(9):
        row = {"Hole": idx + 10, "Par": int(in_pars[idx])}
        for p_i, name in enumerate(players):
            row[name] = int(in_strokes[p_i][idx])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("Hole").reset_index(drop=True)


def has_unknowns_df(df: pd.DataFrame, players) -> bool:
    return (df[players] < 0).any().any()


def totals_check(df: pd.DataFrame, players, totals: dict):
    out_total = totals.get("out_total", [-1]*4)
    in_total = totals.get("in_total", [-1]*4)
    grand_total = totals.get("grand_total", [-1]*4)

    out_sum = [int(df.loc[df["Hole"].between(1, 9), p].sum()) for p in players]
    in_sum = [int(df.loc[df["Hole"].between(10, 18), p].sum()) for p in players]
    grand_sum = [out_sum[i] + in_sum[i] for i in range(4)]

    def candidate_holes(seg_df: pd.DataFrame, player: str, diff: int):
        cand = []
        for _, r in seg_df.iterrows():
            v = int(r[player])
            newv = v - diff
            if 1 <= newv <= 20:
                cand.append(int(r["Hole"]))
        return cand[:6]

    rows, hints = [], []
    for i, p in enumerate(players):
        o_t = int(out_total[i])
        i_t = int(in_total[i])
        g_t = int(grand_total[i])

        o_ok = (o_t == -1) or (o_t == out_sum[i])
        i_ok = (i_t == -1) or (i_t == in_sum[i])
        g_ok = (g_t == -1) or (g_t == grand_sum[i])

        rows.append([
            p,
            out_sum[i], ("" if o_t == -1 else o_t), "OK" if o_ok else "DIFF",
            in_sum[i], ("" if i_t == -1 else i_t), "OK" if i_ok else "DIFF",
            grand_sum[i], ("" if g_t == -1 else g_t), "OK" if g_ok else "DIFF",
        ])

        if o_t != -1 and o_t != out_sum[i]:
            diff = out_sum[i] - o_t
            cand = candidate_holes(df[df["Hole"].between(1, 9)], p, diff)
            hints.append(f"{p} OUT 불일치: (현재 {out_sum[i]})-(카드 {o_t})={diff:+}. 후보 홀: {cand if cand else '없음'}")
        if i_t != -1 and i_t != in_sum[i]:
            diff = in_sum[i] - i_t
            cand = candidate_holes(df[df["Hole"].between(10, 18)], p, diff)
            hints.append(f"{p} IN 불일치: (현재 {in_sum[i]})-(카드 {i_t})={diff:+}. 후보 홀: {cand if cand else '없음'}")
        if g_t != -1 and g_t != grand_sum[i]:
            diff = grand_sum[i] - g_t
            hints.append(f"{p} TOTAL 불일치: (현재 {grand_sum[i]})-(카드 {g_t})={diff:+}. OUT/IN부터 확인 추천")

    df_check = pd.DataFrame(rows, columns=[
        "플레이어",
        "OUT_현재합", "OUT_카드합", "OUT_검증",
        "IN_현재합", "IN_카드합", "IN_검증",
        "TOTAL_현재합", "TOTAL_카드합", "TOTAL_검증",
    ])
    any_diff = (df_check[["OUT_검증", "IN_검증", "TOTAL_검증"]] == "DIFF").any().any()
    return df_check, hints, any_diff


# ======================
# Kevin 룰 (버그 수정 + 버디/이글 1:1 보너스)
# ======================
def label_from_strokes(strokes: int, par: int) -> str:
    diff = strokes - par
    return {
        -2: "이글",
        -1: "버디",
        0: "파",
        1: "보기",
        2: "더블",
        3: "트리플",
        4: "쿼드러플",
    }.get(diff, f"{diff:+}타")


def calculate_hole_kevin(strokes, par, prev_all_tie, base_amount, max_per_stroke):
    n = len(strokes)
    counts = Counter(strokes)
    tie_three = any(v >= 3 for v in counts.values())
    all_tie = len(set(strokes)) == 1
    any_birdie_eagle = any((s - par) <= -1 for s in strokes)

    mult = 1
    reasons = []
    if tie_three:
        mult *= 2
        reasons.append("3명 이상 동타")
    if prev_all_tie:
        mult *= 2
        reasons.append("전홀 동타")
    if any_birdie_eagle:
        mult *= 2
        reasons.append("버디/이글 발생")
    if not reasons:
        reasons.append("없음")

    if all_tie:
        return [0] * n, True, mult, " / ".join(reasons)

    per = base_amount * mult
    if max_per_stroke:
        per = min(per, max_per_stroke)

    labels = [label_from_strokes(s, par) for s in strokes]
    bonus_map = {"이글": -2, "버디": -1}
    adj = [bonus_map.get(lbl, 0) for lbl in labels]
    effective = [strokes[i] + adj[i] for i in range(n)]

    matrix = [[0] * n for _ in range(n)]
    for i, j in combinations(range(n), 2):
        delta = effective[j] - effective[i]
        amt = abs(delta) * per
        if delta > 0:
            matrix[i][j] += amt
            matrix[j][i] -= amt
        elif delta < 0:
            matrix[j][i] += amt
            matrix[i][j] -= amt

    totals = [sum(row) for row in matrix]
    return totals, False, mult, " / ".join(reasons)


# ======================
# Session state
# ======================
if "players" not in st.session_state:
    st.session_state.players = ["P1", "P2", "P3", "P4"]
if "df" not in st.session_state:
    st.session_state.df = None
if "totals_check_df" not in st.session_state:
    st.session_state.totals_check_df = None
if "totals_hints" not in st.session_state:
    st.session_state.totals_hints = None
if "hole_results" not in st.session_state:
    st.session_state.hole_results = None
if "final_results" not in st.session_state:
    st.session_state.final_results = None


# ======================
# Sidebar
# ======================
st.sidebar.header("⚙️ 룰 설정")
base_amount = st.sidebar.number_input("기준금액(타당)", 1000, step=1000, value=5000)
apply_max = st.sidebar.checkbox("타당 최대 금액 적용", value=True)
max_per_stroke = st.sidebar.number_input("타당 최대금액", 1000, step=1000, value=20000) if apply_max else None

st.sidebar.divider()
use_totals_check = st.sidebar.checkbox("OUT/IN 합계 검증(추천)", value=True)


# ======================
# UI
# ======================
uploaded = st.file_uploader("스코어카드 업로드 (한 장에 OUT/IN 포함)", type=["png", "jpg", "jpeg", "webp"])

colA, colB = st.columns(2, gap="large")

with colA:
    st.subheader("1) AI로 스코어 읽기 (2단계 + 재시도)")

    if uploaded:
        img_raw = Image.open(uploaded).convert("RGB")
        img_base = _resize_cap(img_raw, MAX_WIDTH)

        img_light = preprocess_light(img_base)
        st.image(img_light, caption="전처리(기본)", use_container_width=True)

        if st.button("🤖 AI로 읽기"):
            with st.spinner("구조(이름/Par) 추출 중..."):
                struct = extract_structure(img_light)

            players = struct["players"]
            out_pars = struct["out_pars"]
            in_pars = struct["in_pars"]

            with st.spinner("타수 추출(1차) 중..."):
                strokes = extract_strokes(img_light, players, out_pars, in_pars)

            df = to_df18(players, out_pars, in_pars, strokes["out_strokes"], strokes["in_strokes"])

            if has_unknowns_df(df, players):
                st.warning("⚠️ -1이 있어 강전처리로 한 번 더 읽습니다.")
                img_strong = preprocess_strong(img_base)
                st.image(img_strong, caption="전처리(강화/재시도용)", use_container_width=True)

                with st.spinner("타수 재추출(2차/강화) 중..."):
                    strokes2 = extract_strokes(img_strong, players, out_pars, in_pars)

                df2 = to_df18(players, out_pars, in_pars, strokes2["out_strokes"], strokes2["in_strokes"])
                if df2[players].lt(0).sum().sum() < df[players].lt(0).sum().sum():
                    df = df2
                    st.success("✅ 2차 결과가 개선되어 적용했습니다.")
                else:
                    st.info("ℹ️ 2차 결과가 더 좋아지지 않아 1차를 유지합니다.")

            st.session_state.players = players
            st.session_state.df = df
            st.success("✅ 추출 완료! 오른쪽에서 확인/수정 후 정산하세요.")

            # 합계 검증
            st.session_state.totals_check_df = None
            st.session_state.totals_hints = None
            if use_totals_check:
                with st.spinner("합계(OUT/IN/T) 추출 및 검증 중..."):
                    totals = extract_totals(img_light, players)
                check_df, hints, any_diff = totals_check(df, players, totals)
                st.session_state.totals_check_df = check_df
                st.session_state.totals_hints = hints
                if any_diff:
                    st.warning("⚠️ 카드 합계와 현재 입력 합계가 다릅니다. 오른쪽 후보 홀을 먼저 확인해보세요.")
                else:
                    st.success("✅ 합계 검증 OK (또는 카드 합계가 없어 검증 생략됨).")


with colB:
    st.subheader("2) 결과 확인/수정 + 정산(홀별 포함)")

    if st.session_state.df is None:
        st.info("왼쪽에서 AI로 읽기를 실행하세요.")
    else:
        players = st.session_state.players

        if has_unknowns_df(st.session_state.df, players):
            st.warning("⚠️ -1(미확정) 값이 남아있습니다. 수정 후 정산하세요.")

        if st.session_state.totals_check_df is not None:
            st.markdown("### ✅ OUT/IN 합계 검증")
            st.dataframe(st.session_state.totals_check_df, use_container_width=True)
            if st.session_state.totals_hints:
                with st.expander("🔎 틀린 홀 후보(자동 제안)"):
                    for h in st.session_state.totals_hints:
                        st.write("- " + h)

        df_edit = st.data_editor(st.session_state.df, use_container_width=True, num_rows="fixed")
        st.session_state.df = df_edit

        if st.button("💰 18홀 정산 (홀별 내역 포함)"):
            dfv = st.session_state.df.copy()

            for p in players:
                if (dfv[p] < 0).any():
                    st.error("❌ -1 값이 남아있어 정산할 수 없습니다. -1을 모두 수정해주세요.")
                    st.stop()

            prev_all_tie = False
            total = [0, 0, 0, 0]
            hole_rows = []

            for _, r in dfv.sort_values("Hole").iterrows():
                hole = int(r["Hole"])
                par = int(r["Par"])
                strokes = [int(r[p]) for p in players]

                totals, all_tie, mult, reason = calculate_hole_kevin(
                    strokes=strokes,
                    par=par,
                    prev_all_tie=prev_all_tie,
                    base_amount=int(base_amount),
                    max_per_stroke=(int(max_per_stroke) if max_per_stroke else None),
                )

                unit = base_amount * mult
                if max_per_stroke:
                    unit = min(unit, max_per_stroke)

                labels = [label_from_strokes(s, par) for s in strokes]

                for i in range(4):
                    total[i] += totals[i]

                row = {"Hole": hole, "Par": par, "배수": mult, "단가(1타)": unit, "사유": reason}
                for i, name in enumerate(players):
                    row[f"{name}_타수"] = strokes[i]
                    row[f"{name}_라벨"] = labels[i]
                    row[f"{name}_홀정산(+받음/-냄)"] = totals[i]
                hole_rows.append(row)

                prev_all_tie = all_tie

            df_holes = pd.DataFrame(hole_rows).sort_values("Hole")
            st.session_state.hole_results = df_holes

            final_rows = []
            for i, name in enumerate(players):
                amt = int(total[i])
                status = "받음" if amt > 0 else "냄" if amt < 0 else "0원"
                final_rows.append([name, status, f"{abs(amt):,}원", amt])
            df_final = pd.DataFrame(final_rows, columns=["플레이어", "상태", "표시금액", "원본부호금액(+받음/-냄)"])
            st.session_state.final_results = df_final

            st.success("✅ 정산 완료! 아래에서 홀별/최종 결과를 확인하세요.")


# ======================
# Results display
# ======================
if st.session_state.hole_results is not None:
    st.divider()
    st.subheader("📌 홀별 정산 내역 ( +면 받음 / -면 냄 )")
    st.dataframe(st.session_state.hole_results, use_container_width=True)

    st.download_button(
        "⬇️ 홀별 정산 CSV 다운로드",
        data=st.session_state.hole_results.to_csv(index=False).encode("utf-8-sig"),
        file_name="kevin_hole_settlement.csv",
        mime="text/csv",
    )

if st.session_state.final_results is not None:
    st.subheader("🏁 최종 누적 정산")
    st.dataframe(st.session_state.final_results, use_container_width=True)

    st.download_button(
        "⬇️ 최종 정산 CSV 다운로드",
        data=st.session_state.final_results.to_csv(index=False).encode("utf-8-sig"),
        file_name="kevin_final_settlement.csv",
        mime="text/csv",
    )
