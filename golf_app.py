import streamlit as st
import cv2
import numpy as np
import pandas as pd
import easyocr

st.set_page_config(page_title="Golf Scorecard Reader")

st.title("⛳ Golf Scorecard Reader")

# OCR 초기화
reader = easyocr.Reader(['en'])

def preprocess(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray,(5,5),0)

    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2
    )

    return thresh


def read_digit(cell):

    result = reader.readtext(cell, detail=0)

    if len(result) > 0:

        txt = result[0]

        txt = ''.join(filter(str.isdigit, txt))

        if txt.isdigit():
            return int(txt)

    return 0


def extract_scores(img):

    h,w = img.shape

    players = 4
    holes = 18

    start_y = int(h*0.30)
    row_gap = int(h*0.085)

    start_x = int(w*0.20)
    col_gap = int(w*0.040)

    cell_h = int(h*0.06)
    cell_w = int(w*0.035)

    scores=[]

    for p in range(players):

        row=[]

        y = start_y + p*row_gap

        for h_idx in range(holes):

            x = start_x + h_idx*col_gap

            cell = img[y:y+cell_h, x:x+cell_w]

            digit = read_digit(cell)

            row.append(digit)

        scores.append(row)

    return scores


uploaded = st.file_uploader(
    "Upload scorecard screenshot",
    type=["png","jpg","jpeg"]
)

if uploaded:

    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)

    img = cv2.imdecode(file_bytes,1)

    st.image(img, caption="Uploaded Scorecard")

    processed = preprocess(img)

    with st.spinner("Reading scores..."):

        scores = extract_scores(processed)

    players = [
        "Player1",
        "Player2",
        "Player3",
        "Player4"
    ]

    df = pd.DataFrame(
        scores,
        index=players,
        columns=[f"H{i}" for i in range(1,19)]
    )

    st.subheader("Detected Scores")

    edited = st.data_editor(df)

    st.download_button(
        "Download CSV",
        edited.to_csv(),
        "scores.csv"
    )
