import re
import os
import pickle

import pandas as pd
from transformers import pipeline
from stqdm import stqdm
import streamlit as st

import functions as funcs
import preprocess as prep
import mongodb


st.set_page_config(
    page_title="리뷰데이터 제로샷 자연어 추론",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="auto",
)

@st.experimental_memo
def get_df():
    client = mongodb.client
    db_names = mongodb.db_names
    db = client.get_database(db_names[1])
    coll_names = funcs.get_collections(1)
    coll = db[coll_names[5]]

    df = funcs.get_df(coll, 5)

    filename = ['jp_comp_name_list']
    comp_name_ls = tuple(pickle.load(open(filename[0], 'rb')))
    return df, comp_name_ls

@st.experimental_memo
def get_df_by_comp(df, company_name):
    df_comp = funcs.get_comp(df, company_name)
    return df_comp

@st.experimental_memo
def get_df_by_year(df, year):
    df_year = df.query(f'year == {year}')
    return df_year

@st.experimental_memo
def get_model():
    model = pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
    return model

def get_result(_model, docs, candidate_labels, multi_label_input, idx, sample_n):
    multi_label = True if multi_label_input == "ON" else False
    outputs = []
    for doc in stqdm(docs[int(idx):int(idx)+sample_n]):
        output = _model(doc, candidate_labels, multi_label=multi_label)
        outputs.append(output)
    result = pd.DataFrame(outputs)
    result['class'] = result['labels'].apply(lambda x: x[0])
    return result[['sequence', 'class', 'labels', 'scores']]

def get_score_avg_by_label(result):
    dicts = []
    for labels, scores in list(zip(result['labels'].tolist(), result['scores'].tolist())):
        dicts.append(dict(zip(labels, scores)))
    score_df = pd.DataFrame(dicts)
    return score_df.mean()

df, comp_name_ls = get_df()
model = get_model()

with st.sidebar:
    st.text('---[데이터 필터]---')
    year = st.slider(
        '⁜ 연도를 선택하세요.',
        2014, 2022, (2021)
    )
    col = st.selectbox(
        "⁜ 분석 텍스트 필드를 선택하세요.",
        ('장점', '단점', '경영진에게')
    )
    company_name = st.selectbox(
        "⁜ 회사명을 입력/선택하세요.",
        comp_name_ls
    )

st.title('[그레이비랩 기업부설 연구소 / AI lab.]')
st.subheader(f'{year}년 {company_name}')
with st.container():
    default_candidate_labels = ['복지 및 급여', '워라밸', '사내문화', '승진 기회 및 가능성']
    user_input = st.text_input(
    f"✓ 사용자 레이블을 입력하시고, 콤마로 분리하세요. (default={default_candidate_labels})",
    ""
    )
    if user_input:
        candidate_labels = [x.strip() for x in user_input.split(',')]
    else:
        candidate_labels = default_candidate_labels


    col1, col2, col3, col4, col5 = st.columns([5,1,5,1,5])
    with col1:
        idx = st.text_input(
            "✓ 조회할 데이터 시작 인덱스를 입력하세요. (defalut=0)",
            ""
        )
    with col3:
        sample_n = st.slider(
            "✓ 조회할 데이터 총 개수를 선택하세요.",
            1, 100, (10)
        )
    with col5:
        multi_label_input = st.radio(
            "✓ 멀티 레이블을 키고 끌 수 있습니다.",
            ('On', 'Off')
        )

    if not idx:
        idx = 0

df_comp = get_df_by_comp(df, company_name)
df_year = get_df_by_year(df_comp, year)

col_dic = {'장점': 'Pros', '단점': 'Cons', '경영진에게': 'To_Management'}

st.subheader("Result")
col1, col2 = st.columns([7, 1, 2])
with col1:
    docs = df_year[col_dic[col]].apply(prep.preprocess_text).tolist()
    result = get_result(model, docs, candidate_labels, multi_label_input, idx, sample_n)
    st.dataframe(result)
    st.caption(f"{year}년 {company_name}추론 결과표")

with col2:
    score_avg = get_score_avg_by_label(result)
    st.dataframe(score_avg)
    st.caption(f"{year}년 {company_name} 각 레이블 평균 추론 스코어")

with st.expander("⁜ 자세히 보기 : 사용한 DL model - [mDeBERTa-v3-base-xnli-multilingual-nli-2mil7]"):
    st.markdown(
        """
이 다국어 모델은 100개 언어에 대해 자연어 추론(NLI)을 수행할 수 있으므로 다국어 제로샷 분류에도 적합합니다. 기본 mDeBERTa-v3-base 모델은 100개 언어로 구성된 CC100 다국어 데이터 세트에서 Microsoft에 의해 사전 훈련되었습니다. 그런 다음 모델은 XNLI 데이터 세트와 다국어 NLI-26lang-2mil7 데이터 세트에서 fine-tune되었습니다. 두 데이터 세트 모두 40억 명이 넘는 사람들이 사용하는 27개 언어로 된 270만 개 이상의 가설-전제 쌍을 포함합니다.
        """
    )
