import streamlit as st
import cv2
import numpy as np
import pandas as pd
import pytesseract
from PIL import Image

st.title("골프 스코어카드 인식기")

uploaded_file = st.file_uploader("스코어카드 캡처 업로드", type=["png","jpg","jpeg"])

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    th = cv2.adaptiveThreshold(
        blur,255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,2
    )
    return th

def read_digit(cell):

    config = "--psm 10 -c tessedit_char_whitelist=0123456789"

    txt = pytesseract.image_to_string(cell, config=config)

    txt = txt.strip()

    if txt == "":
        return 0

    try:
        return int(txt)
    except:
        return 0


def extract_scores(img):

    h, w = img.shape[:2]

    start_x = int(w*0.18)
    start_y = int(h*0.40)

    cell_w = int(w*0.045)
    cell_h = int(h*0.035)

    gap_x = int(w*0.048)
    gap_y = int(h*0.040)

    scores = []

    for player in range(4):

        row = []

        for hole in range(18):

            x = start_x + hole*gap_x
            y = start_y + player*gap_y

            cell = img[y:y+cell_h, x:x+cell_w]

            proc = preprocess(cell)

            digit = read_digit(proc)

            row.append(digit)

        scores.append(row)

    return scores


if uploaded_file:

    image = Image.open(uploaded_file)
    img = np.array(image)

    st.image(image, caption="업로드된 이미지")

    scores = extract_scores(img)

    df = pd.DataFrame(
        scores,
        columns=[f"{i}H" for i in range(1,19)],
        index=[f"Player{i+1}" for i in range(4)]
    )

    st.subheader("인식 결과")

    st.dataframe(df)
