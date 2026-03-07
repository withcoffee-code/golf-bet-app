import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("⛳ Golf Scorecard AI Reader")

uploaded = st.file_uploader("스코어카드 업로드", type=["png","jpg","jpeg"])

VALID_VALUES = [-1,0,1,2,3]

def normalize_score(v):

    try:
        v = int(v)
    except:
        return 0

    if v in VALID_VALUES:
        return v

    # 자동 보정
    if v > 3:
        return 3

    if v < -1:
        return -1

    return v


def read_cell(cell):

    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(gray,(60,60))

    _,th = cv2.threshold(gray,120,255,cv2.THRESH_BINARY_INV)

    pixels = np.sum(th)/255

    # 픽셀 기반 단순 추정
    if pixels < 300:
        return 0

    if pixels < 600:
        return 1

    if pixels < 900:
        return 2

    return 3


def extract_scores(img):

    img = cv2.resize(img,(1170,2532))

    h,w,_ = img.shape

    rows = 4
    cols = 18

    start_x = int(w*0.18)
    start_y = int(h*0.39)

    table_w = int(w*0.72)
    table_h = int(h*0.34)

    cell_w = table_w // cols
    cell_h = table_h // rows

    scores = []

    for r in range(rows):

        row = []

        for c in range(cols):

            x = start_x + c*cell_w
            y = start_y + r*cell_h

            cell = img[y:y+cell_h,x:x+cell_w]

            val = read_cell(cell)

            val = normalize_score(val)

            row.append(val)

        scores.append(row)

    return scores


if uploaded:

    image = Image.open(uploaded)
    img = np.array(image)

    st.image(img)

    scores = extract_scores(img)

    st.subheader("읽은 스코어")

    for i,row in enumerate(scores):

        st.write(f"Player {i+1}",row)
