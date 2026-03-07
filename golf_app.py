import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("⛳ Golf Scorecard Reader")

uploaded = st.file_uploader("스코어카드 업로드", type=["png","jpg","jpeg"])

# 숫자 템플릿 생성
def generate_templates():

    templates = {}

    for i in range(10):

        img = np.zeros((60,40),dtype=np.uint8)

        cv2.putText(
            img,
            str(i),
            (5,50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.8,
            255,
            3
        )

        templates[i] = img

    return templates


templates = generate_templates()


def read_digit(cell):

    cell = cv2.resize(cell,(40,60))

    best = None
    best_score = -1

    for digit,temp in templates.items():

        res = cv2.matchTemplate(cell,temp,cv2.TM_CCOEFF_NORMED)
        score = res[0][0]

        if score > best_score:

            best_score = score
            best = digit

    return best


def preprocess(cell):

    if cell is None or cell.size == 0:
        return None

    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray,(3,3),0)

    _,th = cv2.threshold(
        blur,
        120,
        255,
        cv2.THRESH_BINARY_INV
    )

    return th


def extract_scores(img):

    # 항상 같은 크기로 맞춤
    img = cv2.resize(img,(1170,2532))

    h,w,_ = img.shape

    rows = 4
    cols = 9

    start_x = int(w*0.22)
    start_y = int(h*0.40)

    table_w = int(w*0.65)
    table_h = int(h*0.33)

    cell_w = table_w // cols
    cell_h = table_h // rows

    scores = []

    for r in range(rows):

        row = []

        for c in range(cols):

            x = start_x + c*cell_w
            y = start_y + r*cell_h

            cell = img[y:y+cell_h,x:x+cell_w]

            th = preprocess(cell)

            if th is None:
                row.append(0)
                continue

            digit = read_digit(th)

            row.append(digit)

        scores.append(row)

    return scores


if uploaded:

    image = Image.open(uploaded)
    img = np.array(image)

    st.image(img, caption="업로드 이미지")

    scores = extract_scores(img)

    st.subheader("인식 결과")

    for i,row in enumerate(scores):

        st.write(f"Player {i+1}",row)
