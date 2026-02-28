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
st.title("⛳ Kevin 룰 계산기")

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
# Image utils
# ======================
def _resize_cap(img: Image.Image, max_w=MAX_WIDTH) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    if w > max_w:
        ratio = max_w / w
        img = img.resize((int(w * ratio), int(h * ratio)), Image.Resampling.LANCZOS)
    return img


def auto_crop_table(img: Image.Image, pad: int = 18) -> Image.Image:
    """표/숫자 영역만 자동 크롭(PIL만 사용)"""
    base = _resize_cap(img, MAX_WIDTH)
    gray = ImageOps.grayscale(base)
    gray = ImageOps.autocontrast(gray)

    # 어두운 픽셀 강조
    bw = gray.point(lambda p: 255 if p < 200 else 0)
    bbox = bw.getbbox()
    if not bbox:
        return base

    x0, y0, x1, y1 = bbox
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(base.size[0], x1 + pad)
    y1 = min(base.size[1], y1 + pad)
    return base.crop((x0, y0, x1, y1))


def find_split_y(img: Image.Image) -> int | None:
    """
    OUT/IN 표가 위/아래로 나뉜 경우가 많아서,
    '어두운 픽셀 밀도'가 낮은 수평 구간(빈 줄)을 찾아 분할선으로 사용.
    """
    base = _resize_cap(img, MAX_WIDTH)
    gray = ImageOps.grayscale(base)
    gray = ImageOps.autocontrast(gray)

    w, h = gray.size
    # 글자/선(어두운 픽셀)을 1로
    bw = gray.point(lambda p: 1 if p < 200 else 0)

    # row density
    row_sum = [0] * h
    px = bw.load()
    for y in range(h):
        s = 0
        for x in range(w):
            s += px[x, y]
        row_sum[y] = s

    # 가운데 근처에서 "골" 찾기 (중간 +/- 25%)
    y_start = int(h * 0.30)
    y_end = int(h * 0.70)
    if y_end - y_start < 50:
        return None

    window = 25  # smoothing
    smooth = [0] * h
    for y in range(h):
        a = max(0, y - window)
        b = min(h, y + window + 1)
        smooth[y] = sum(row_sum[a:b]) / (b - a)

    # 중앙 밸리 후보: 최소값
    valley_y = min(range(y_start, y_end), key=lambda y: smooth[y])

    # 너무 위/아래면 실패 처리
    if valley_y < int(h * 0.25) or valley_y > int(h * 0.75):
        return None

    return valley_y


def split_out_in(img: Image.Image, pad: int = 10):
    """자동 분할선 기반으로 OUT/IN 이미지 2개 반환. 실패 시 상/하 반반."""
    base = _resize_cap(img, MAX_WIDTH)
    w, h = base.size
    y = find_split_y(base)
    if y is None:
        y = h // 2

    y0 = max(0, y - pad)
    y1 = min(h, y + pad)

    out_img = base.crop((0, 0, w, y1))
    in_img = base.crop((0, y0, w, h))
    return out_img, in_img, y


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
    # 보수적 이진화
    gray = gray.point(lambda p: 255 if p > 200 else 0)
    return gray.convert("RGB")


def to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


# ======================
# JSON Schemas (strict)
# ======================
PLAYERS_SCHEMA = {
    "name": "scorecard_players",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "players": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 4,
                "maxItems": 4,
            }
        },
        "required": ["players"],
    },
    "strict": True,
}

PARS9_SCHEMA = {
    "name": "scorecard_pars9",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "pars": {
                "type": "array",
                "items": {"type": "integer", "enum": [3, 4, 5]},
                "minItems": 9,
                "maxItems": 9,
            }
        },
        "required": ["pars"],
    },
    "strict": True,
}

STROKES9_SCHEMA = {
    "name": "scorecard_strokes9",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "strokes": {
                "type": "array",
                "minItems": 4,
                "maxItems": 4,
                "items": {
                    "type": "array",
                    "minItems": 9,
                    "maxItems": 9,
                    "items": {"type": "integer", "minimum": -1, "maximum": 20},
                },
            }
        },
        "required": ["strokes"],
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
# Extraction (players from full, pars/strokes from split)
# ======================
def extract_players(img_full: Image.Image):
    prompt = """
이 이미지는 골프 스코어카드이다.
반드시 players 4명의 이름만 추출해라. (표에서 타수가 기록된 순서대로)
JSON 외 텍스트 출력 금지.
"""
    content = [{"type": "input_text", "text": prompt}, {"type": "input_image", "image_url": to_data_url(img_full)}]
    return call_json_schema(content, PLAYERS_SCHEMA)["players"]


def extract_pars9(img_seg: Image.Image, segment_name: str):
    prompt = f"""
이 이미지는 스코어카드의 {segment_name} 9홀 표(Par 행/열 포함)이다.
반드시 PAR 9개만 순서대로 추출해라.
JSON 외 텍스트 출력 금지.
"""
    content = [{"type": "input_text", "text": prompt}, {"type": "input_image", "image_url": to_data_url(img_seg)}]
    return call_json_schema(content, PARS9_SCHEMA)["pars"]


def extract_strokes9(img_seg: Image.Image, players, pars9, segment_name: str):
    prompt = f"""
이 이미지는 스코어카드의 {segment_name} 9홀 표이다.
players 순서는 고정이며 절대 변경하지 마라.

players: {players}
pars(9개): {pars9}

너는 이제 타수 숫자만 읽어라.
- strokes: players 순서대로 각 플레이어의 타수 9개 (총 4x9)

규칙:
- 확실히 못 읽으면 -1 (추정 금지)
- 아이콘/동그라미/색칠은 숫자 아님
- JSON 외 출력 금지
"""
    content = [{"type": "input_text", "text": prompt}, {"type": "input_image", "image_url": to_data_url(img_seg)}]
    return call_json_schema(content, STROKES9_SCHEMA)["strokes"]


def extract_totals(img_full: Image.Image, players):
    prompt = f"""
이 이미지는 골프 스코어카드이다.
players 순서는 고정이다.

players: {players}

카드에 OUT/IN/TOTAL 합계 칸이 있으면 읽어라:
- out_total / in_total / grand_total (각 4명)
없거나 확실히 못 읽으면 -1.
추정 금지. JSON 외 출력 금지.
"""
    content = [{"type": "input_text", "text": prompt}, {"type": "input_image", "image_url": to_data_url(img_full)}]
    return call_json_schema(content, TOTALS_SCHEMA)


# ======================
# DF build + checks
# ======================
def to_df18(players, out_pars, in_pars, out_strokes, in_strokes):
    rows = []
    for i in range(9):
        r = {"Hole": i + 1, "Par": int(out_pars[i])}
        for pi, p in enumerate(players):
            r[p] = int(out_strokes[pi][i])
        rows.append(r)
    for i in range(9):
        r = {"Hole": i + 10, "Par": int(in_pars[i])}
        for pi, p in enumerate(players):
            r[p] = int(in_strokes[pi][i])
        rows.append(r)
    return pd.DataFrame(rows).sort_values("Hole").reset_index(drop=True)


def has_unknowns_4x9(strokes_4x9):
    return any(v < 0 for row in strokes_4x9 for v in row)


def has_unknowns_df(df: pd.DataFrame, players):
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

    rows = []
    hints = []
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
    """
    totals[i] > 0 => i가 받음
    totals[i] < 0 => i가 냄
    """
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
use_autocrop = st.sidebar.checkbox("표 영역 자동 크롭(추천)", value=True)
use_split = st.sidebar.checkbox("OUT/IN 자동 2분할(추천)", value=True)
use_totals_check = st.sidebar.checkbox("OUT/IN 합계 검증(추천)", value=True)


# ======================
# UI
# ======================
uploaded = st.file_uploader("스코어카드 업로드 (한 장에 OUT/IN 포함)", type=["png", "jpg", "jpeg", "webp"])

colA, colB = st.columns(2, gap="large")

with colA:
    st.subheader("1) AI로 스코어 읽기 (크롭 + 2분할 + 세그먼트 재시도)")

    if uploaded:
        img_raw = Image.open(uploaded).convert("RGB")

        # 크롭
        img_base = auto_crop_table(img_raw) if use_autocrop else _resize_cap(img_raw, MAX_WIDTH)

        # 2분할
        if use_split:
            out_img, in_img, split_y = split_out_in(img_base)
            st.caption(f"자동 분할선 y={split_y}")
        else:
            out_img, in_img = img_base, img_base  # (비권장) 동일 이미지로 처리

        # 전처리(기본)
        out_light = preprocess_light(out_img)
        in_light = preprocess_light(in_img)
        full_light = preprocess_light(img_base)

        st.image(full_light, caption="전처리(전체/이름&합계용)", use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            st.image(out_light, caption="OUT 표(추정) 전처리", use_container_width=True)
        with c2:
            st.image(in_light, caption="IN 표(추정) 전처리", use_container_width=True)

        if st.button("🤖 AI로 읽기"):
            with st.spinner("플레이어 이름 추출(전체 이미지) 중..."):
                players = extract_players(full_light)

            with st.spinner("OUT PAR 추출(OUT 표) 중..."):
                out_pars = extract_pars9(out_light, "OUT(1~9)")

            with st.spinner("IN PAR 추출(IN 표) 중..."):
                in_pars = extract_pars9(in_light, "IN(10~18)")

            # OUT 타수 1차
            with st.spinner("OUT 타수 추출(1차) 중..."):
                out_strokes = extract_strokes9(out_light, players, out_pars, "OUT(1~9)")
            # OUT -1이면 OUT만 강전처리 재시도
            if has_unknowns_4x9(out_strokes):
                st.warning("⚠️ OUT 타수에 -1이 있어 OUT 표만 강전처리 재시도")
                out_strong = preprocess_strong(out_img)
                st.image(out_strong, caption="OUT 표 강전처리", use_container_width=True)
                with st.spinner("OUT 타수 재추출(강화) 중..."):
                    out_strokes2 = extract_strokes9(out_strong, players, out_pars, "OUT(1~9)")
                if sum(v < 0 for row in out_strokes2 for v in row) < sum(v < 0 for row in out_strokes for v in row):
                    out_strokes = out_strokes2
                    st.success("✅ OUT 재시도 결과를 적용했습니다.")

            # IN 타수 1차
            with st.spinner("IN 타수 추출(1차) 중..."):
                in_strokes = extract_strokes9(in_light, players, in_pars, "IN(10~18)")
            # IN -1이면 IN만 강전처리 재시도
            if has_unknowns_4x9(in_strokes):
                st.warning("⚠️ IN 타수에 -1이 있어 IN 표만 강전처리 재시도")
                in_strong = preprocess_strong(in_img)
                st.image(in_strong, caption="IN 표 강전처리", use_container_width=True)
                with st.spinner("IN 타수 재추출(강화) 중..."):
                    in_strokes2 = extract_strokes9(in_strong, players, in_pars, "IN(10~18)")
                if sum(v < 0 for row in in_strokes2 for v in row) < sum(v < 0 for row in in_strokes for v in row):
                    in_strokes = in_strokes2
                    st.success("✅ IN 재시도 결과를 적용했습니다.")

            df = to_df18(players, out_pars, in_pars, out_strokes, in_strokes)
            st.session_state.players = players
            st.session_state.df = df

            st.success("✅ 추출 완료! 오른쪽에서 확인/수정 후 정산하세요.")

            # 합계 검증
            st.session_state.totals_check_df = None
            st.session_state.totals_hints = None
            if use_totals_check:
                with st.spinner("합계(OUT/IN/T) 추출 및 검증 중..."):
                    totals = extract_totals(full_light, players)
                check_df, hints, any_diff = totals_check(df, players, totals)
                st.session_state.totals_check_df = check_df
                st.session_state.totals_hints = hints
                if any_diff:
                    st.warning("⚠️ 카드 합계와 현재 입력 합계가 다릅니다. 오른쪽의 후보 홀을 먼저 확인해보세요.")
                else:
                    st.success("✅ 합계 검증 OK (또는 카드 합계가 없어 검증 생략됨).")


with colB:
    st.subheader("2) 결과 확인/수정 + 정산(홀별 포함)")

    if st.session_state.df is None:
        st.info("왼쪽에서 AI로 읽기를 실행하세요.")
    else:
        players = st.session_state.players

        if has_unknowns_df(st.session_state.df, players):
            st.warning("⚠️ -1(미확정) 값이 남아있습니다. 해당 칸을 수정해야 정산이 정확합니다.")

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
