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


# ----------------------
# 기본 설정
# ----------------------
st.set_page_config(page_title="Kevin 룰 계산기", layout="wide")
st.title("⛳ Kevin 룰 계산기 (최종 안정 버전)")

MAX_WIDTH = 1800


# ----------------------
# API 키 처리
# ----------------------
def get_api_key():
    try:
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except:
        pass
    return os.getenv("OPENAI_API_KEY")


def get_client():
    key = get_api_key()
    if not key:
        st.error("❌ OPENAI_API_KEY가 설정되지 않았습니다.")
        st.stop()
    return OpenAI(api_key=key)


# ----------------------
# 이미지 전처리
# ----------------------
def preprocess_image(img):
    img = img.convert("RGB")
    w, h = img.size

    if w > MAX_WIDTH:
        ratio = MAX_WIDTH / w
        img = img.resize((int(w * ratio), int(h * ratio)), Image.Resampling.LANCZOS)

    gray = ImageOps.grayscale(img)
    gray = ImageEnhance.Contrast(gray).enhance(1.3)

    return gray.convert("RGB")


def to_data_url(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ----------------------
# JSON 스키마 (strict 완전 대응)
# ----------------------
SCORECARD_SCHEMA = {
    "name": "scorecard_out_in",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "players": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 4,
                "maxItems": 4
            },
            "out": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "pars": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 9,
                        "maxItems": 9
                    },
                    "strokes": {
                        "type": "array",
                        "minItems": 4,
                        "maxItems": 4,
                        "items": {
                            "type": "array",
                            "items": {
                                "type": "integer",
                                "minimum": -1,
                                "maximum": 20
                            },
                            "minItems": 9,
                            "maxItems": 9
                        }
                    }
                },
                "required": ["pars", "strokes"]
            },
            "in": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "pars": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 9,
                        "maxItems": 9
                    },
                    "strokes": {
                        "type": "array",
                        "minItems": 4,
                        "maxItems": 4,
                        "items": {
                            "type": "array",
                            "items": {
                                "type": "integer",
                                "minimum": -1,
                                "maximum": 20
                            },
                            "minItems": 9,
                            "maxItems": 9
                        }
                    }
                },
                "required": ["pars", "strokes"]
            }
        },
        "required": ["players", "out", "in"]
    },
    "strict": True
}


# ----------------------
# AI 호출 (자동 fallback)
# ----------------------
def extract_scorecard(img):

    client = get_client()

    content = [{
        "type": "input_text",
        "text": """
이 이미지는 골프 스코어카드이다.

- OUT(1~9)과 IN(10~18)을 분리해서 읽어라.
- 숫자를 확실히 읽을 수 없으면 -1을 써라.
- 추정하지 마라.
- JSON 외 텍스트 출력 금지.
"""
    }, {
        "type": "input_image",
        "image_url": to_data_url(img)
    }]

    try:
        resp = client.responses.create(
            model="gpt-4.1",
            input=[{"role": "user", "content": content}],
            text={
                "format": {
                    "type": "json_schema",
                    "name": SCORECARD_SCHEMA["name"],
                    "schema": SCORECARD_SCHEMA["schema"],
                    "strict": True
                }
            }
        )
        return json.loads(resp.output_text)

    except Exception as e:
        st.warning("⚠ gpt-4.1 실패 → gpt-4.1-mini 재시도")
        try:
            resp = client.responses.create(
                model="gpt-4.1-mini",
                input=[{"role": "user", "content": content}],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": SCORECARD_SCHEMA["name"],
                        "schema": SCORECARD_SCHEMA["schema"],
                        "strict": True
                    }
                }
            )
            return json.loads(resp.output_text)
        except Exception as e2:
            st.error("❌ OpenAI 호출 실패")
            st.code(str(e2))
            st.stop()


# ----------------------
# OUT/IN → 18홀 변환
# ----------------------
def convert_to_df(data):
    players = data["players"]
    rows = []

    for i in range(9):
        row = {"Hole": i + 1, "Par": data["out"]["pars"][i]}
        for p_i, p in enumerate(players):
            row[p] = data["out"]["strokes"][p_i][i]
        rows.append(row)

    for i in range(9):
        row = {"Hole": i + 10, "Par": data["in"]["pars"][i]}
        for p_i, p in enumerate(players):
            row[p] = data["in"]["strokes"][p_i][i]
        rows.append(row)

    return pd.DataFrame(rows)


# ----------------------
# Kevin 룰 계산
# ----------------------
def calculate_hole(strokes, par, prev_all_tie, base, max_cap):

    n = 4
    counts = Counter(strokes)

    tie_three = any(v >= 3 for v in counts.values())
    all_tie = len(set(strokes)) == 1
    birdie = any((s - par) <= -1 for s in strokes)

    mult = 1
    if tie_three: mult *= 2
    if prev_all_tie: mult *= 2
    if birdie: mult *= 2

    per = base * mult
    if max_cap:
        per = min(per, max_cap)

    matrix = [[0] * n for _ in range(n)]

    for i, j in combinations(range(n), 2):
        diff = strokes[j] - strokes[i]
        amt = abs(diff) * per

        if diff > 0:
            matrix[i][j] += amt
            matrix[j][i] -= amt
        elif diff < 0:
            matrix[j][i] += amt
            matrix[i][j] -= amt

    totals = [sum(row) for row in matrix]
    return totals, all_tie


# ----------------------
# UI
# ----------------------
st.sidebar.header("Kevin 룰 설정")
base = st.sidebar.number_input("기준금액", 1000, value=5000, step=1000)
max_cap = st.sidebar.number_input("최대 타당 금액", 1000, value=20000, step=1000)

uploaded = st.file_uploader("스코어카드 업로드", type=["png","jpg","jpeg","webp"])

if uploaded and st.button("🤖 AI로 읽기"):

    img = Image.open(uploaded)
    img = preprocess_image(img)
    st.image(img, use_container_width=True)

    data = extract_scorecard(img)
    df = convert_to_df(data)

    st.session_state.players = data["players"]
    st.session_state.df = df

    if (df[st.session_state.players] < 0).any().any():
        st.warning("⚠ 확신 못한 숫자(-1)가 있습니다. 수정하세요.")


if "df" in st.session_state:

    df = st.data_editor(st.session_state.df, use_container_width=True)
    st.session_state.df = df

    if st.button("💰 정산"):

        prev = False
        total = [0,0,0,0]

        for _, r in df.iterrows():
            strokes = [int(r[p]) for p in st.session_state.players]
            totals, prev = calculate_hole(strokes, int(r["Par"]), prev, base, max_cap)
            for i in range(4):
                total[i] += totals[i]

        result = []
        for i, p in enumerate(st.session_state.players):
            amt = total[i]
            result.append([p, "받음" if amt > 0 else "냄", f"{abs(amt):,}원"])

        st.dataframe(pd.DataFrame(result, columns=["플레이어","상태","금액"]))
