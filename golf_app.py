import streamlit as st
import easyocr
import numpy as np
import pandas as pd
from PIL import Image

# 페이지 설정
st.set_page_config(page_title="골프 스코어카드 리더", page_icon="⛳")

st.title("⛳ 골프 스코어카드 리더")
st.write("카카오골프예약 앱의 스코어카드 스크린샷을 업로드하세요.")

# 세션 상태에서 OCR 모델 로드 (캐싱을 통해 속도 향상)
@st.cache_resource
def load_ocr_model():
    # 한글과 숫자를 인식하도록 설정
    return easyocr.Reader(['ko', 'en'])

reader = load_ocr_model()

# 파일 업로드
uploaded_file = st.file_uploader("이미지 파일 업로드 (JPG, PNG)", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # 이미지 표시
    image = Image.open(uploaded_file)
    st.image(image, caption='업로드된 스코어카드', use_container_width=True)
    
    if st.button('스코어 분석 시작'):
        with st.spinner('이미지에서 데이터를 읽어오는 중입니다...'):
            # 1. OCR 인식
            img_array = np.array(image)
            results = reader.readtext(img_array)
            
            # 2. 데이터 정리 (간단한 파싱 로직)
            # 여기서는 모든 텍스트를 리스트로 추출한 뒤, 
            # 실제 서비스시에는 좌표나 특정 키워드(HOLE, PAR)를 기준으로 필터링하는 로직이 추가됩니다.
            
            data = []
            for (bbox, text, prob) in results:
                if prob > 0.4:  # 신뢰도 40% 이상인 데이터만 수집
                    data.append(text)
            
            # 3. 결과 표시
            st.success('분석 완료!')
            
            # 대략적인 이름 추출 예시 (이미지 구조상 상단 혹은 표 좌측에 위치)
            # 카카오 양식의 경우 '웰링턴' 같은 골프장 이름과 플레이어 이름이 인식됩니다.
            st.subheader("📋 추출된 데이터 요약")
            
            # 인식된 전체 텍스트를 표 형태로 보여주기 (디버깅 및 확인용)
            df = pd.DataFrame(data, columns=["추출된 텍스트"])
            st.dataframe(df, use_container_width=True)
            
            # 4. (심화) 특정 플레이어 스코어 매칭 가이드
            st.info("""
            **Tip:** 현재는 전체 텍스트를 순서대로 나열합니다. 
            정확한 행/열 매칭을 위해서는 좌표(bbox) 정보를 활용하여 
            같은 높이(y축)에 있는 텍스트를 한 줄로 묶는 로직을 추가하면 
            완벽한 스코어보드를 복구할 수 있습니다.
            """)

# 배포를 위한 안내
with st.sidebar:
    st.header("배포 가이드")
    st.markdown("""
    1. **GitHub**에 이 코드를 `app.py`로 저장하세요.
    2. 같은 폴더에 `requirements.txt`와 `packages.txt`를 만드세요.
    3. **Streamlit Cloud**에서 해당 레포지토리를 연결하세요.
    """)
