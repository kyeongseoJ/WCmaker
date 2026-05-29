import io
import re
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from wordcloud import WordCloud, STOPWORDS


st.set_page_config(
    page_title="WordCloud Generator",
    page_icon="☁️",
    layout="wide",
)


def read_table(uploaded_file):
    """CSV 또는 Excel 파일을 DataFrame으로 읽기"""
    filename = uploaded_file.name.lower()

    if filename.endswith(".csv"):
        try:
            return pd.read_csv(uploaded_file)
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding="cp949")

    if filename.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)

    raise ValueError("CSV 또는 Excel 파일만 지원합니다.")


def make_mask(image_file, invert_mask=False, threshold=245):
    """
    업로드한 이미지를 WordCloud mask로 변환.
    WordCloud에서는 흰색 영역이 단어가 배치되지 않는 영역으로 처리됩니다.
    """
    image = Image.open(image_file).convert("RGBA")

    # 투명 배경이 있는 PNG도 처리
    background = Image.new("RGBA", image.size, (255, 255, 255, 255))
    background.paste(image, mask=image.split()[3])
    image = background.convert("RGB")

    gray = image.convert("L")
    arr = np.array(gray)

    # 밝은 영역은 흰색, 어두운 영역은 단어가 들어갈 영역으로 사용
    mask = np.where(arr > threshold, 255, 0).astype(np.uint8)

    if invert_mask:
        mask = 255 - mask

    return mask, image


def build_text_from_column(df, text_column, min_word_len, custom_stopwords):
    series = df[text_column].dropna().astype(str)

    text = " ".join(series.tolist())

    # 너무 짧은 단어 제거
    words = re.findall(r"\b[\w가-힣]+\b", text)
    words = [
        word
        for word in words
        if len(word) >= min_word_len and word not in custom_stopwords
    ]

    return " ".join(words)


def build_frequencies(df, word_column, freq_column, min_word_len, custom_stopwords):
    temp = df[[word_column, freq_column]].dropna().copy()
    temp[word_column] = temp[word_column].astype(str)
    temp[freq_column] = pd.to_numeric(temp[freq_column], errors="coerce")
    temp = temp.dropna()

    temp = temp[
        temp[word_column].str.len().ge(min_word_len)
        & ~temp[word_column].isin(custom_stopwords)
    ]

    return dict(zip(temp[word_column], temp[freq_column]))


def wordcloud_to_png_bytes(wc):
    img = wc.to_image()
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


st.title("☁️ 이미지 모양 워드클라우드 생성기")
st.caption("CSV 또는 Excel 데이터를 넣고, 원하는 이미지 모양으로 워드클라우드를 생성합니다.")

with st.sidebar:
    st.header("⚙️ 워드클라우드 설정")

    data_mode = st.radio(
        "데이터 형식",
        ["텍스트 컬럼 사용", "단어-빈도 컬럼 사용"],
        help="텍스트 컬럼은 문장/리뷰/댓글 데이터에 적합하고, 단어-빈도 컬럼은 이미 집계된 데이터에 적합합니다.",
    )

    max_words = st.slider("최대 단어 수", 20, 1000, 200, 10)
    min_word_len = st.slider("최소 단어 길이", 1, 10, 2)
    min_font_size = st.slider("최소 글자 크기", 1, 50, 4)
    max_font_size = st.slider("최대 글자 크기", 20, 300, 120)

    background_color = st.color_picker("배경색", "#FFFFFF")
    contour_color = st.color_picker("외곽선 색상", "#333333")
    contour_width = st.slider("외곽선 두께", 0, 20, 2)

    colormap = st.selectbox(
        "컬러맵",
        [
            "viridis",
            "plasma",
            "inferno",
            "magma",
            "cividis",
            "tab10",
            "Set2",
            "Pastel1",
            "cool",
            "hot",
            "rainbow",
        ],
    )

    prefer_horizontal = st.slider(
        "가로 배치 비율",
        0.0,
        1.0,
        0.9,
        0.05,
        help="1에 가까울수록 단어가 가로로 많이 배치됩니다.",
    )

    relative_scaling = st.slider(
        "빈도 기반 크기 반영",
        0.0,
        1.0,
        0.5,
        0.05,
        help="0이면 순위 중심, 1이면 빈도 차이를 더 강하게 반영합니다.",
    )

    random_state = st.number_input("랜덤 시드", min_value=0, max_value=9999, value=42)

    st.divider()

    mask_threshold = st.slider(
        "마스크 밝기 기준",
        0,
        255,
        245,
        help="이미지에서 밝은 영역과 어두운 영역을 나누는 기준입니다.",
    )
    invert_mask = st.checkbox(
        "마스크 반전",
        value=False,
        help="단어가 들어가는 영역이 반대로 보이면 켜세요.",
    )

    st.divider()

    font_path = st.text_input(
        "폰트 파일 경로",
        value="NanumGothic.ttf",
        help="한국어는 반드시 한글을 지원하는 TTF/OTF 폰트 경로가 필요합니다.",
    )

    custom_stopwords_text = st.text_area(
        "제외할 단어",
        value="그리고\n그러나\n하지만\n저는\n제가\n입니다",
        help="한 줄에 하나씩 입력하세요.",
    )

custom_stopwords = {
    word.strip()
    for word in custom_stopwords_text.splitlines()
    if word.strip()
}
stopwords = set(STOPWORDS).union(custom_stopwords)

left, right = st.columns([1, 1])

with left:
    st.subheader("1. 데이터 업로드")
    data_file = st.file_uploader(
        "CSV 또는 Excel 파일을 업로드하세요.",
        type=["csv", "xlsx", "xls"],
    )

    st.subheader("2. 모양 이미지 업로드")
    image_file = st.file_uploader(
        "워드클라우드 모양으로 사용할 이미지를 업로드하세요.",
        type=["png", "jpg", "jpeg"],
    )

df = None

if data_file is not None:
    try:
        df = read_table(data_file)
        with left:
            st.success("데이터를 불러왔습니다.")
            st.dataframe(df.head(20), use_container_width=True)
    except Exception as e:
        st.error(f"데이터 파일을 읽는 중 오류가 발생했습니다: {e}")

if image_file is not None:
    try:
        mask, original_image = make_mask(
            image_file,
            invert_mask=invert_mask,
            threshold=mask_threshold,
        )

        with right:
            st.subheader("마스크 미리보기")
            preview_col1, preview_col2 = st.columns(2)
            with preview_col1:
                st.image(original_image, caption="원본 이미지", use_container_width=True)
            with preview_col2:
                st.image(mask, caption="워드클라우드 마스크", use_container_width=True)

    except Exception as e:
        st.error(f"이미지 파일을 처리하는 중 오류가 발생했습니다: {e}")
        mask = None
else:
    mask = None

if df is not None:
    st.divider()
    st.subheader("3. 컬럼 선택")

    columns = df.columns.tolist()

    if data_mode == "텍스트 컬럼 사용":
        text_column = st.selectbox("텍스트가 들어있는 컬럼", columns)
        freq_column = None
    else:
        col1, col2 = st.columns(2)
        with col1:
            text_column = st.selectbox("단어 컬럼", columns)
        with col2:
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            default_freq_index = (
                columns.index(numeric_cols[0]) if numeric_cols else 0
            )
            freq_column = st.selectbox(
                "빈도/가중치 컬럼",
                columns,
                index=default_freq_index,
            )

    generate_button = st.button("워드클라우드 생성", type="primary")

    if generate_button:
        try:
            if data_mode == "텍스트 컬럼 사용":
                text = build_text_from_column(
                    df=df,
                    text_column=text_column,
                    min_word_len=min_word_len,
                    custom_stopwords=custom_stopwords,
                )

                if not text.strip():
                    st.warning("워드클라우드로 만들 텍스트가 없습니다.")
                    st.stop()

                generate_input = text
                use_frequencies = False

            else:
                frequencies = build_frequencies(
                    df=df,
                    word_column=text_column,
                    freq_column=freq_column,
                    min_word_len=min_word_len,
                    custom_stopwords=custom_stopwords,
                )

                if not frequencies:
                    st.warning("워드클라우드로 만들 단어-빈도 데이터가 없습니다.")
                    st.stop()

                generate_input = frequencies
                use_frequencies = True

            wc = WordCloud(
                font_path=font_path,
                mask=mask,
                max_words=max_words,
                min_font_size=min_font_size,
                max_font_size=max_font_size,
                background_color=background_color,
                contour_width=contour_width if mask is not None else 0,
                contour_color=contour_color,
                colormap=colormap,
                prefer_horizontal=prefer_horizontal,
                relative_scaling=relative_scaling,
                random_state=random_state,
                stopwords=stopwords,
                width=1000,
                height=700,
                collocations=False,
            )

            if use_frequencies:
                wc.generate_from_frequencies(generate_input)
            else:
                wc.generate(generate_input)

            st.divider()
            st.subheader("4. 결과")

            result_image = wc.to_image()
            st.image(result_image, use_container_width=True)

            png_bytes = wordcloud_to_png_bytes(wc)
            st.download_button(
                label="PNG 다운로드",
                data=png_bytes,
                file_name="wordcloud.png",
                mime="image/png",
            )

            with st.expander("상위 단어 확인"):
                word_freq = wc.words_
                top_words = pd.DataFrame(
                    word_freq.items(),
                    columns=["word", "relative_frequency"],
                )
                st.dataframe(top_words.head(100), use_container_width=True)

        except OSError:
            st.error(
                "폰트 파일을 찾을 수 없습니다. 사이드바의 폰트 파일 경로를 확인하세요. "
                "한국어 데이터는 NanumGothic.ttf 같은 한글 지원 폰트가 필요합니다."
            )
        except Exception as e:
            st.error(f"워드클라우드 생성 중 오류가 발생했습니다: {e}")