import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

st.title("Golf Scorecard Reader")

# MNIST CNN 모델 로드
model = tf.keras.models.load_model(
    tf.keras.utils.get_file(
        "mnist_model.h5",
        "https://storage.googleapis.com/tensorflow/keras-datasets/mnist_model.h5"
    )
)

def predict_digit(cell):

    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)

    _,th = cv2.threshold(gray,150,255,cv2.THRESH_BINARY_INV)

    resized = cv2.resize(th,(28,28))

    norm = resized/255.0

    norm = norm.reshape(1,28,28,1)

    pred = model.predict(norm,verbose=0)

    digit = np.argmax(pred)

    if digit>4:
        digit = digit%4

    return digit


def extract_scores(img,start_y):

    h,w,_ = img.shape

    score_x1 = int(w*0.25)

    col_w = int(w*0.065)

    row_h = int(h*0.045)

    scores=[]

    for p in range(4):

        row=[]

        y1 = start_y + (p+2)*row_h

        y2 = y1 + row_h

        for h_idx in range(9):

            x1 = score_x1 + h_idx*col_w

            x2 = x1 + col_w

            cell = img[y1:y2,x1:x2]

            digit = predict_digit(cell)

            row.append(digit)

        scores.append(row)

    return scores


def parse_scorecard(img):

    h,w,_ = img.shape

    out_start = int(h*0.32)

    in_start = int(h*0.57)

    out_scores = extract_scores(img,out_start)

    in_scores = extract_scores(img,in_start)

    final=[]

    for i in range(4):

        final.append(out_scores[i]+in_scores[i])

    players=["김경만","허균","홍성완","이기원"]

    return final,players


uploaded = st.file_uploader("Upload Scorecard")

if uploaded:

    file_bytes = np.asarray(bytearray(uploaded.read()),dtype=np.uint8)

    img = cv2.imdecode(file_bytes,1)

    st.image(img)

    scores,players = parse_scorecard(img)

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
