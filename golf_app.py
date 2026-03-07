import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image

st.set_page_config(page_title="Golf Scorecard Parser")
st.title("⛳ 카카오 골프 스코어카드 파서")

uploaded = st.file_uploader("스코어카드 업로드", type=["png","jpg","jpeg"])

def preprocess(cell):
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(3,3),0)
    _,th = cv2.threshold(blur,150,255,cv2.THRESH_BINARY_INV)
    return th

def classify_digit(img):

    h,w = img.shape

    area = np.sum(img==255)

    if area < 20:
        return 0

    left = np.sum(img[:,0:int(w*0.4)]==255)
    right = np.sum(img[:,int(w*0.6):]==255)
    top = np.sum(img[0:int(h*0.4),:]==255)
    bottom = np.sum(img[int(h*0.6):,:]==255)

    if left < 5 and right > 10:
        return 1

    if top > bottom and left > right:
        return 2

    if right > left and top > 5:
        return 3

    return 4


def extract_scores(img):

    h,w,_ = img.shape

    start_x = int(w*0.23)
    start_y = int(h*0.40)

    cell_w = int(w*0.045)
    cell_h = int(h*0.035)

    gap_x = int(w*0.048)
    gap_y = int(h*0.040)

    scores=[]

    for p in range(4):

        row=[]

        for hole in range(18):

            x = start_x + hole*gap_x
            y = start_y + p*gap_y

            cell = img[y:y+cell_h, x:x+cell_w]

            th = preprocess(cell)

            digit = classify_digit(th)

            row.append(digit)

        scores.append(row)

    return scores


if uploaded:

    image = Image.open(uploaded)
    img = np.array(image)

    st.image(image)

    scores = extract_scores(img)

    df = pd.DataFrame(
        scores,
        columns=[f"H{i}" for i in range(1,19)],
        index=["Player1","Player2","Player3","Player4"]
    )

    st.subheader("인식 결과")

    edited = st.data_editor(df)

    st.download_button(
        "CSV 다운로드",
        edited.to_csv(),
        "scores.csv"
    )
