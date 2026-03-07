import streamlit as st
import cv2
import numpy as np
import pandas as pd
import easyocr

st.title("Golf Scorecard Reader")

reader = easyocr.Reader(['en','ko'])


def remove_icons(img):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    yellow = cv2.inRange(hsv,(20,100,100),(35,255,255))

    img[yellow>0] = (255,255,255)

    return img


def read_digit(cell):

    result = reader.readtext(cell, detail=0)

    if result:

        txt = result[0]

        txt = ''.join(filter(str.isdigit, txt))

        if txt.isdigit():
            return int(txt)

    return 0


def extract_table(img, start_y):

    h,w,_ = img.shape

    name_x1 = int(w*0.05)
    name_x2 = int(w*0.22)

    score_x1 = int(w*0.25)
    score_x2 = int(w*0.90)

    row_h = int(h*0.045)
    col_w = int((score_x2-score_x1)/10)

    players=[]
    scores=[]

    for i in range(4):

        y1 = start_y + (i+2)*row_h
        y2 = y1 + row_h

        row=[]

        for j in range(9):

            x1 = score_x1 + j*col_w
            x2 = x1 + col_w

            cell = img[y1:y2, x1:x2]

            digit = read_digit(cell)

            row.append(digit)

        scores.append(row)

    return scores


def parse_scorecard(img):

    img = remove_icons(img)

    h,w,_ = img.shape

    out_start = int(h*0.32)
    in_start = int(h*0.57)

    out_scores = extract_table(img,out_start)
    in_scores = extract_table(img,in_start)

    final=[]

    for i in range(4):

        final.append(out_scores[i]+in_scores[i])

    players=["김경만","허균","홍성완","이기원"]

    return final, players


uploaded = st.file_uploader("Upload Scorecard")

if uploaded:

    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)

    img = cv2.imdecode(file_bytes,1)

    st.image(img)

    scores, players = parse_scorecard(img)

    df = pd.DataFrame(
        scores,
        index=players,
        columns=[f"H{i}" for i in range(1,19)]
    )

    edited = st.data_editor(df)

    st.download_button(
        "Download CSV",
        edited.to_csv(),
        "scores.csv"
    )
