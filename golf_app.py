# app.py
# Kevin 룰 계산기 - OUT/IN 통합 카드 + 숫자 오독 개선 풀옵션 버전

import os
import base64
import io
import json
from collections import Counter
from itertools import combinations

import pandas as pd
import streamlit as st
from PIL import Image, ImageEnhance, ImageOps

try:
    import pypdfium2 as pdfium
    PDFIUM_AVAILABLE = True
except:
    PDFIUM_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except:
    OPENAI_AVAILABLE = False


# ----------------------
# 페이지 설정
# ----------------------
st.set_page_config(page_title="Kevin 룰 계산기", layout="wide")
st.title("⛳ Kevin 룰 계산기 (AI 스코어카드 자동정산 - 강화버전)")


# ----------------------
# API KEY 처리
# ----------------------
def get_openai_api_key():
    try:
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except:
        pass
    return os.getenv("OPENAI_API_KEY")

def get_client():
    key = get_openai_api_key()
    if not key:
        raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다.")
    return OpenAI(api_key=key)

api_key_present = get_openai_api_key() is not None


# ----------------------
# 세션 초기화
# ----------------------
if "players" not in st.session_state:
    st.session_state.players = ["P1","P2","P3","P4"]
if "base_amount" not in st.session_state:
    st.session_state.base_amount = 5000
if "max_per_stroke" not in st.session_state:
    st.session_state.max_per_stroke = 20000
if "apply_max_toggle" not in st.session_state:
    st.session_state.apply_max_toggle = True
if "edited_df" not in st.session_state:
    st.session_state.edited_df = None
if "calc_history" not in st.session_state:
    st.session_state.calc_history = []
if "calc_total" not in st.session_state:
    st.session_state.calc_total = [0,0,0,0]


# ----------------------
# 사이드바
# ----------------------
st.sidebar.header("⚙️ Kevin 룰 설정")

st.session_state.base_amount = st.sidebar.number_input(
    "기준금액 (타당)", 1000, step=1000, value=st.session_state.base_amount
)

st.session_state.apply_max_toggle = st.sidebar.checkbox(
    "타당 최대 금액 적용", value=st.session_state.apply_max_toggle
)

if st.session_state.apply_max_toggle:
    st.session_state.max_per_stroke = st.sidebar.number_input(
        "타당 최대 금액", 1000, step=1000, value=st.session_state.max_per_stroke
    )
else:
    st.session_state.max_per_stroke = None

st.sidebar.divider()

# 🔥 OCR 보정 옵션
st.sidebar.subheader("🧪 숫자 인식 보정")
upscale = st.sidebar.selectbox("업스케일", [1.0,1.5,2.0,2.5], index=2)
contrast = st.sidebar.selectbox("대비", [1.0,1.3,1.6,2.0], index=2)
sharpness = st.sidebar.selectbox("선명도", [1.0,1.2,1.3,1.5], index=2)

st.sidebar.caption("값을 올리면 숫자 오독이 줄어들 수 있습니다.")


# ----------------------
# 이미지 전처리
# ----------------------
def preprocess_for_ocr(img):
    img = img.convert("RGB")
    if upscale != 1.0:
        w,h = img.size
        img = img.resize((int(w*upscale), int(h*upscale)), Image.Resampling.LANCZOS)

    gray = ImageOps.grayscale(img)
    gray = ImageOps.autocontrast(gray)
    gray = ImageEnhance.Contrast(gray).enhance(contrast)
    gray = ImageEnhance.Sharpness(gray).enhance(sharpness)

    return gray.convert("RGB")


def pil_to_data_url(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ----------------------
# AI 스키마 (OUT/IN 분리 + -1 허용)
# ----------------------
SCORECARD_SCHEMA = {
    "name": "scorecard_out_in",
    "schema": {
        "type": "object",
        "properties": {
            "players": {
                "type": "array",
                "items": {"type":"string"},
                "minItems":4,"maxItems":4
            },
            "out": {
                "type":"object",
                "properties":{
                    "pars":{"type":"array","items":{"type":"integer"},"minItems":9,"maxItems":9},
                    "strokes":{
                        "type":"array",
                        "minItems":4,"maxItems":4,
                        "items":{
                            "type":"array",
                            "items":{"type":"integer","minimum":-1,"maximum":20},
                            "minItems":9,"maxItems":9
                        }
                    }
                },
                "required":["pars","strokes"]
            },
            "in": {
                "type":"object",
                "properties":{
                    "pars":{"type":"array","items":{"type":"integer"},"minItems":9,"maxItems":9},
                    "strokes":{
                        "type":"array",
                        "minItems":4,"maxItems":4,
                        "items":{
                            "type":"array",
                            "items":{"type":"integer","minimum":-1,"maximum":20},
                            "minItems":9,"maxItems":9
                        }
                    }
                },
                "required":["pars","strokes"]
            }
        },
        "required":["players","out","in"]
    },
    "strict": True
}


def extract_scorecard(image):
    client = get_client()

    content = [{
        "type":"input_text",
        "text":"""
이 이미지는 골프 스코어카드이다.

- OUT(1~9)과 IN(10~18)을 분리해서 읽어라.
- pars는 정확히 9개씩.
- strokes는 플레이어 4명 각각 9개씩.
- 숫자를 확실히 읽을 수 없으면 추정하지 말고 -1을 써라.
- 별표/동그라미/아이콘은 무시하라.
- JSON 외 텍스트 출력 금지.
"""
    },{
        "type":"input_image",
        "image_url": pil_to_data_url(image)
    }]

    resp = client.responses.create(
        model="gpt-4.1",
        input=[{"role":"user","content":content}],
        text={
            "format":{
                "type":"json_schema",
                "name":SCORECARD_SCHEMA["name"],
                "schema":SCORECARD_SCHEMA["schema"],
                "strict":True
            }
        }
    )

    return json.loads(resp.output_text)


# ----------------------
# OUT/IN → 18홀 DF 변환
# ----------------------
def out_in_to_df(data):
    players = data["players"]
    rows=[]
    for i in range(9):
        row={"Hole":i+1,"Par":data["out"]["pars"][i]}
        for p_i,p in enumerate(players):
            row[p]=data["out"]["strokes"][p_i][i]
        rows.append(row)
    for i in range(9):
        row={"Hole":i+10,"Par":data["in"]["pars"][i]}
        for p_i,p in enumerate(players):
            row[p]=data["in"]["strokes"][p_i][i]
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------
# Kevin 룰 계산
# ----------------------
def calculate_hole(strokes,par,prev_all_tie):
    n=4
    counts=Counter(strokes)
    tie_three=any(v>=3 for v in counts.values())
    all_tie=len(set(strokes))==1
    birdie=any((s-par)<=-1 for s in strokes)

    mult=1
    if tie_three: mult*=2
    if prev_all_tie: mult*=2
    if birdie: mult*=2

    per=st.session_state.base_amount*mult
    if st.session_state.max_per_stroke:
        per=min(per,st.session_state.max_per_stroke)

    matrix=[[0]*n for _ in range(n)]
    for i,j in combinations(range(n),2):
        diff=strokes[j]-strokes[i]
        amt=abs(diff)*per
        if diff>0:
            matrix[i][j]+=amt
            matrix[j][i]-=amt
        elif diff<0:
            matrix[j][i]+=amt
            matrix[i][j]-=amt

    totals=[sum(row) for row in matrix]
    return totals,all_tie


# ----------------------
# UI
# ----------------------
if not api_key_present:
    st.warning("OPENAI_API_KEY가 설정되지 않았습니다.")

uploaded=st.file_uploader("스코어카드 업로드",type=["png","jpg","jpeg","webp","pdf"])

if uploaded and st.button("🤖 AI로 읽기"):
    if uploaded.type=="application/pdf":
        pdf=pdfium.PdfDocument(uploaded.getvalue())
        img=pdf[0].render(scale=2).to_pil()
    else:
        img=Image.open(uploaded)

    img=preprocess_for_ocr(img)
    st.image(img,caption="전처리 이미지",use_container_width=True)

    data=extract_scorecard(img)
    st.session_state.players=data["players"]
    df=out_in_to_df(data)
    st.session_state.edited_df=df

    if (df[st.session_state.players]<0).any().any():
        st.warning("⚠️ AI가 확신 못한 숫자(-1)가 있습니다. 수정해주세요.")

# ----------------------
# 편집 & 정산
# ----------------------
if st.session_state.edited_df is not None:
    df=st.data_editor(st.session_state.edited_df,use_container_width=True)
    st.session_state.edited_df=df

    if st.button("💰 18홀 정산"):
        prev=False
        total=[0,0,0,0]
        history=[]
        for _,r in df.iterrows():
            strokes=[int(r[p]) for p in st.session_state.players]
            totals,prev=calculate_hole(strokes,int(r["Par"]),prev)
            for i in range(4):
                total[i]+=totals[i]
        st.session_state.calc_total=total

        result=[]
        for i,p in enumerate(st.session_state.players):
            amt=total[i]
            status="받음" if amt>0 else "냄"
            result.append([p,status,f"{abs(amt):,}원"])
        st.dataframe(pd.DataFrame(result,columns=["플레이어","상태","금액"]))
