# app.py
# Streamlit: 스코어카드(사진/PDF) 업로드 → AI Vision으로 (플레이어 이름 + OUT/IN 9홀씩) 추출 →
# (필요 시 수정) → Kevin 룰로 18홀 일괄 정산
#
# 필요:
#   pip install streamlit pandas pillow openai pypdfium2
# 실행(로컬):
#   export OPENAI_API_KEY="..."
#   streamlit run app.py
#
# 실행(Streamlit Cloud):
#   Settings > Secrets 에 아래 추가
#   OPENAI_API_KEY="sk-..."

import os
import base64
import io
import json
from collections import Counter
from itertools import combinations

import pandas as pd
import streamlit as st
from PIL import Image

# PDF -> 이미지 변환
try:
    import pypdfium2 as pdfium
    PDFIUM_AVAILABLE = True
except Exception:
    PDFIUM_AVAILABLE = False

# OpenAI
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False


# ----------------------
# 페이지 설정
# ----------------------
st.set_page_config(page_title="Kevin 룰 계산기 (스코어카드 AI 일괄정산)", layout="wide")
st.title("⛳ Kevin 룰 계산기 (스코어카드 AI 일괄정산) — OUT/IN 통합 카드 지원")


# ----------------------
# API KEY 체크/클라이언트
# ----------------------
def get_openai_api_key() -> str | None:
    # 1) Streamlit Cloud secrets
    try:
        if "OPENAI_API_KEY" in st.secrets and st.secrets["OPENAI_API_KEY"]:
            return str(st.secrets["OPENAI_API_KEY"]).strip()
    except Exception:
        pass

    # 2) 환경변수
    key = os.getenv("OPENAI_API_KEY")
    if key and key.strip():
        return key.strip()

    return None


def get_openai_client():
    if not OPENAI_AVAILABLE:
        raise RuntimeError("OpenAI SDK가 필요합니다. (pip install openai)")
    api_key = get_openai_api_key()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다.")
    return OpenAI(api_key=api_key)


api_key_present = get_openai_api_key() is not None


# ----------------------
# 세션 상태 초기화
# ----------------------
if "players" not in st.session_state:
    st.session_state.players = ["PLAYER1", "PLAYER2", "PLAYER3", "PLAYER4"]
if "base_amount" not in st.session_state:
    st.session_state.base_amount = 5000
if "apply_max_toggle" not in st.session_state:
    st.session_state.apply_max_toggle = True
if "max_per_stroke" not in st.session_state:
    st.session_state.max_per_stroke = 20000
if "extracted_df" not in st.session_state:
    st.session_state.extracted_df = None
if "edited_df" not in st.session_state:
    st.session_state.edited_df = None
if "calc_history" not in st.session_state:
    st.session_state.calc_history = []
if "calc_total" not in st.session_state:
    st.session_state.calc_total = [0, 0, 0, 0]
if "raw_ai_json" not in st.session_state:
    st.session_state.raw_ai_json = None


# ----------------------
# 룰 설정(사이드바)
# ----------------------
st.sidebar.header("⚙️ 룰 설정")
st.session_state.base_amount = st.sidebar.number_input(
    "기준금액 (타당)", min_value=1000, step=1000, value=int(st.session_state.base_amount)
)

st.session_state.apply_max_toggle = st.sidebar.checkbox(
    "타당 최대 금액 적용", value=bool(st.session_state.apply_max_toggle)
)

if st.session_state.apply_max_toggle:
    st.session_state.max_per_stroke = st.sidebar.number_input(
        "타당 최대 금액 (1타 기준)", min_value=1000, step=1000, value=int(st.session_state.max_per_stroke)
    )
else:
    st.session_state.max_per_stroke = None

st.sidebar.divider()
st.sidebar.caption("✅ 정산 부호: + 받음 / - 냄")
st.sidebar.caption("✅ 배판: (3명 이상 동타 / 전홀 동타 / 버디·이글 발생) 각각 2배씩 곱")
st.sidebar.caption("✅ 버디/이글 보너스: 1:1에서 버디=-1타, 이글=-2타 유효타수로 반영")
st.sidebar.caption("✅ OUT/IN이 한 장에 같이 있는 스코어카드에 최적화")


# ----------------------
# 유틸: 이미지/스키마/비전 호출
# ----------------------
def pil_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def pdf_to_images(file_bytes: bytes):
    if not PDFIUM_AVAILABLE:
        raise RuntimeError("PDF 처리를 위해 pypdfium2 설치가 필요합니다. (pip install pypdfium2)")
    pdf = pdfium.PdfDocument(file_bytes)
    images = []
    for i in range(len(pdf)):
        page = pdf[i]
        pil = page.render(scale=2).to_pil()
        images.append(pil.convert("RGB"))
    return images


# ✅ OUT/IN 분리 스키마 (한 장에 OUT/IN 모두 있을 때 정확도↑)
SCORECARD_SCHEMA = {
    "name": "scorecard_out_in",
    "schema": {
        "type": "object",
        "properties": {
            "players": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 4,
                "maxItems": 4
            },
            "out": {
                "type": "object",
                "properties": {
                    "pars": {
                        "type": "array",
                        "items": {"type": "integer", "enum": [3, 4, 5]},
                        "minItems": 9,
                        "maxItems": 9
                    },
                    "strokes": {
                        "type": "array",
                        "minItems": 4,
                        "maxItems": 4,
                        "items": {
                            "type": "array",
                            "items": {"type": "integer", "minimum": 1, "maximum": 20},
                            "minItems": 9,
                            "maxItems": 9
                        }
                    }
                },
                "required": ["pars", "strokes"],
                "additionalProperties": False
            },
            "in": {
                "type": "object",
                "properties": {
                    "pars": {
                        "type": "array",
                        "items": {"type": "integer", "enum": [3, 4, 5]},
                        "minItems": 9,
                        "maxItems": 9
                    },
                    "strokes": {
                        "type": "array",
                        "minItems": 4,
                        "maxItems": 4,
                        "items": {
                            "type": "array",
                            "items": {"type": "integer", "minimum": 1, "maximum": 20},
                            "minItems": 9,
                            "maxItems": 9
                        }
                    }
                },
                "required": ["pars", "strokes"],
                "additionalProperties": False
            }
        },
        "required": ["players", "out", "in"],
        "additionalProperties": False
    },
    "strict": True
}


def extract_scorecard_from_images(images_pil, model_name: str):
    client = get_openai_client()

    content = [{"type": "input_text", "text": """
너는 골프 스코어카드 비전 추출기다. 이 이미지는 한 장 안에 OUT(1~9)과 IN(10~18)이 함께 있다.

반드시 아래를 추출해라:
1) players: 4명 이름을 스코어 표의 기록 순서대로.
2) out.pars: OUT 1~9홀의 PAR 9개
3) out.strokes: players 순서대로 각 플레이어의 OUT 타수 9개 (총 4개 배열)
4) in.pars: IN 10~18홀의 PAR 9개
5) in.strokes: players 순서대로 각 플레이어의 IN 타수 9개 (총 4개 배열)

규칙:
- OUT과 IN을 절대 섞지 마라.
- pars는 정확히 9개, strokes도 각 플레이어당 9개여야 한다.
- players 순서와 strokes 인덱스는 1:1 대응.
- JSON 스키마 외 텍스트는 절대 출력하지 마라.
"""}]

    for img in images_pil:
        content.append({"type": "input_image", "image_url": pil_to_data_url(img)})

    # ✅ 최소 패치 유지: text.format.name/schema/strict
    resp = client.responses.create(
        model=model_name,
        input=[{"role": "user", "content": content}],
        text={
            "format": {
                "type": "json_schema",
                "name": SCORECARD_SCHEMA["name"],
                "schema": SCORECARD_SCHEMA["schema"],
                "strict": True,
            }
        },
    )
    return resp.output_text


def out_in_to_df(data: dict) -> pd.DataFrame:
    players = data["players"]
    rows = []

    # OUT: 1~9
    out_pars = data["out"]["pars"]
    out_strokes = data["out"]["strokes"]  # shape: [4][9]
    for idx in range(9):
        row = {"Hole": idx + 1, "Par": int(out_pars[idx])}
        for p_i, name in enumerate(players):
            row[name] = int(out_strokes[p_i][idx])
        rows.append(row)

    # IN: 10~18
    in_pars = data["in"]["pars"]
    in_strokes = data["in"]["strokes"]  # shape: [4][9]
    for idx in range(9):
        row = {"Hole": idx + 10, "Par": int(in_pars[idx])}
        for p_i, name in enumerate(players):
            row[name] = int(in_strokes[p_i][idx])
        rows.append(row)

    return pd.DataFrame(rows).sort_values("Hole").reset_index(drop=True)


# ----------------------
# Kevin 룰 계산(버그 수정 버전)
# ----------------------
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


def calculate_hole_from_strokes(strokes, par, prev_all_tie, base_amount, max_per_stroke):
    n = len(strokes)

    counts = Counter(strokes)
    tie_three = any(v >= 3 for v in counts.values())
    all_tie = len(set(strokes)) == 1
    any_birdie_eagle = any((s - par) <= -1 for s in strokes)

    batch_multiplier = 1
    reasons = []
    if tie_three:
        batch_multiplier *= 2
        reasons.append("3명 이상 동타 → 배판")
    if prev_all_tie:
        batch_multiplier *= 2
        reasons.append("전홀 동타 → 배판")
    if any_birdie_eagle:
        batch_multiplier *= 2
        reasons.append("버디/이글 발생 → 배판")
    if not reasons:
        reasons.append("배판 없음")

    reason_str = "\n".join(reasons)

    if all_tie:
        money_matrix = [[0] * n for _ in range(n)]
        return [0] * n, money_matrix, True, reason_str, batch_multiplier

    per_stroke_amount = base_amount * batch_multiplier
    if max_per_stroke:
        per_stroke_amount = min(per_stroke_amount, max_per_stroke)

    labels = [label_from_strokes(s, par) for s in strokes]
    bonus_map = {"이글": -2, "버디": -1}
    adj = [bonus_map.get(lbl, 0) for lbl in labels]
    effective = [strokes[i] + adj[i] for i in range(n)]

    money_matrix = [[0] * n for _ in range(n)]
    for i, j in combinations(range(n), 2):
        delta = effective[j] - effective[i]  # +면 j가 못침 -> j가 i에게 냄
        amt = abs(delta) * per_stroke_amount

        if delta > 0:
            money_matrix[i][j] += amt
            money_matrix[j][i] -= amt
        elif delta < 0:
            money_matrix[j][i] += amt
            money_matrix[i][j] -= amt

    totals = [sum(row) for row in money_matrix]
    return totals, money_matrix, False, reason_str, batch_multiplier


def validate_df18(df: pd.DataFrame, players: list[str]) -> pd.DataFrame:
    required = ["Hole", "Par"] + players
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")

    out = df.copy()
    out["Hole"] = pd.to_numeric(out["Hole"], errors="coerce").astype("Int64")
    out["Par"] = pd.to_numeric(out["Par"], errors="coerce").astype("Int64")
    for p in players:
        out[p] = pd.to_numeric(out[p], errors="coerce").astype("Int64")

    if out[required].isna().any().any():
        raise ValueError("Hole/Par/타수 중 숫자로 변환 불가한 값이 있습니다.")

    holes = sorted(out["Hole"].astype(int).tolist())
    if holes != list(range(1, 19)):
        raise ValueError(f"Hole 값이 1~18 연속이 아닙니다. 현재: {holes}")

    if not out["Par"].astype(int).isin([3, 4, 5]).all():
        raise ValueError("Par는 3/4/5만 허용합니다.")

    return out.sort_values("Hole").reset_index(drop=True)


# ----------------------
# 상단 안내: API 키 상태 표시
# ----------------------
if not OPENAI_AVAILABLE:
    st.error("OpenAI SDK(openai)가 설치되어 있지 않습니다. `pip install openai` 후 다시 실행하세요.")
elif not api_key_present:
    st.warning(
        "OpenAI API 키가 설정되지 않았습니다.\n\n"
        "• 로컬: 터미널에서 `export OPENAI_API_KEY=\"sk-...\"` 후 실행\n"
        "• Streamlit Cloud: Settings → Secrets 에 `OPENAI_API_KEY=\"sk-...\"` 추가"
    )


# ----------------------
# UI: 업로드 + AI 추출
# ----------------------
left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("📷 스코어카드 업로드")
    uploaded = st.file_uploader("사진(PNG/JPG/WEBP) 또는 PDF", type=["png", "jpg", "jpeg", "webp", "pdf"])

    st.caption("팁: OUT/IN이 한 장에 같이 있는 전체 스코어카드가 가장 잘 인식됩니다.")

    model_name = st.text_input("Vision 모델", value="gpt-4.1-mini")

    images = []
    if uploaded:
        if uploaded.type == "application/pdf":
            try:
                images = pdf_to_images(uploaded.getvalue())
            except Exception as e:
                st.error(f"PDF 처리 오류: {e}")
        else:
            try:
                images = [Image.open(uploaded).convert("RGB")]
            except Exception as e:
                st.error(f"이미지 열기 오류: {e}")

    if images:
        st.image(images[0], caption="미리보기(첫 페이지/첫 이미지)", use_container_width=True)
        if len(images) > 1:
            st.caption(f"총 {len(images)}페이지/이미지")

    col_a, col_b = st.columns(2)
    with col_a:
        read_btn = st.button(
            "🤖 AI로 스코어카드 읽기",
            disabled=(not images) or (not OPENAI_AVAILABLE) or (not api_key_present)
        )
    with col_b:
        reset_btn = st.button("🔄 결과 리셋")

    if reset_btn:
        st.session_state.extracted_df = None
        st.session_state.edited_df = None
        st.session_state.calc_history = []
        st.session_state.calc_total = [0, 0, 0, 0]
        st.session_state.raw_ai_json = None
        st.success("리셋 완료")

    if read_btn:
        try:
            with st.spinner("AI가 스코어카드를 읽는 중..."):
                raw = extract_scorecard_from_images(images, model_name=model_name)
            st.session_state.raw_ai_json = raw
            data = json.loads(raw)

            # players 자동 반영 + OUT/IN -> 18홀 DF 변환
            st.session_state.players = data["players"]
            df18 = out_in_to_df(data)

            st.session_state.extracted_df = df18
            st.session_state.edited_df = df18.copy()
            st.success("AI 추출 완료! 오른쪽에서 이름/타수를 확인·수정한 뒤 정산하세요.")

        except Exception as e:
            st.error(f"AI 추출 오류: {e}")

    if st.session_state.raw_ai_json:
        with st.expander("🔍 AI 원본 JSON 보기"):
            st.code(st.session_state.raw_ai_json, language="json")


with right:
    st.subheader("✍️ 추출 결과 확인 / 수정")

    if st.session_state.extracted_df is None:
        st.info("왼쪽에서 스코어카드를 업로드하고 “AI로 읽기”를 눌러주세요.")
    else:
        st.caption("플레이어 이름이 틀리면 여기서 수정하세요. (컬럼도 자동 반영됩니다)")

        ai_names = st.session_state.players
        n1 = st.text_input("플레이어 1", ai_names[0], key="name_0")
        n2 = st.text_input("플레이어 2", ai_names[1], key="name_1")
        n3 = st.text_input("플레이어 3", ai_names[2], key="name_2")
        n4 = st.text_input("플레이어 4", ai_names[3], key="name_3")
        new_names = [n1, n2, n3, n4]

        df = st.session_state.edited_df.copy()
        score_cols = [c for c in df.columns if c not in ["Hole", "Par"]]
        if score_cols != new_names:
            rename_map = {score_cols[i]: new_names[i] for i in range(4)}
            df = df.rename(columns=rename_map)
            st.session_state.players = new_names
            st.session_state.edited_df = df

        edited = st.data_editor(
            st.session_state.edited_df,
            use_container_width=True,
            num_rows="fixed"
        )
        st.session_state.edited_df = edited

        st.divider()

        calc_btn = st.button("💰 18홀 한 번에 정산")
        if calc_btn:
            try:
                players = st.session_state.players
                df_valid = validate_df18(st.session_state.edited_df, players)

                prev_all_tie = False
                total = [0, 0, 0, 0]
                history = []

                for _, r in df_valid.iterrows():
                    hole = int(r["Hole"])
                    par = int(r["Par"])
                    strokes = [int(r[p]) for p in players]

                    totals, matrix, all_tie, reason, mult = calculate_hole_from_strokes(
                        strokes=strokes,
                        par=par,
                        prev_all_tie=prev_all_tie,
                        base_amount=int(st.session_state.base_amount),
                        max_per_stroke=(int(st.session_state.max_per_stroke) if st.session_state.max_per_stroke else None),
                    )

                    labels = [label_from_strokes(s, par) for s in strokes]
                    for i in range(4):
                        total[i] += totals[i]

                    row = {
                        "Hole": hole,
                        "Par": par,
                        "배수": mult,
                        "배판사유": reason.replace("\n", " / "),
                    }
                    for i, name in enumerate(players):
                        row[f"{name}_타수"] = strokes[i]
                        row[f"{name}_라벨"] = labels[i]
                        row[f"{name}_홀정산(+받음/-냄)"] = totals[i]
                    history.append(row)

                    prev_all_tie = all_tie

                st.session_state.calc_total = total
                st.session_state.calc_history = history
                st.success("정산 완료! 아래 결과를 확인하세요.")

            except Exception as e:
                st.error(f"정산 오류: {e}")


# ----------------------
# 결과 표시
# ----------------------
if st.session_state.calc_history:
    st.divider()
    st.subheader("📌 홀별 정산 결과 ( +면 받음 / -면 냄 )")
    df_h = pd.DataFrame(st.session_state.calc_history).sort_values("Hole")
    st.dataframe(df_h, use_container_width=True)

    st.subheader("🏁 최종 누적 정산")
    players = st.session_state.players
    final_rows = []
    for i, name in enumerate(players):
        amt = int(st.session_state.calc_total[i])
        status = "받음" if amt > 0 else "냄" if amt < 0 else "0원"
        final_rows.append([name, status, f"{abs(amt):,}원", amt])
    df_final = pd.DataFrame(final_rows, columns=["플레이어", "상태", "표시금액", "원본부호금액(+받음/-냄)"])
    st.dataframe(df_final, use_container_width=True)

    st.download_button(
        "⬇️ 홀별 정산 CSV 다운로드",
        data=df_h.to_csv(index=False).encode("utf-8-sig"),
        file_name="kevin_hole_settlement.csv",
        mime="text/csv"
    )
    st.download_button(
        "⬇️ 최종 정산 CSV 다운로드",
        data=df_final.to_csv(index=False).encode("utf-8-sig"),
        file_name="kevin_final_settlement.csv",
        mime="text/csv"
    )
