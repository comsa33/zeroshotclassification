import re
import os
import pickle

import pandas as pd
from transformers import pipeline
import torch
from stqdm import stqdm
import streamlit as st

import functions as funcs
import preprocess as prep
import mongodb


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
pd.set_option('display.max_colwidth', -1)

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

@st.experimental_memo
def get_result(_model, docs, candidate_labels, multi_label_input):
    multi_label = True if multi_label_input == "ON" else False
    outputs = []
    for doc in stqdm(docs):
        output = _model(doc, candidate_labels, multi_label=multi_label)
        outputs.append(output)
    result = pd.DataFrame(outputs)
    result['class'] = result['labels'].apply(lambda x: x[0])
    return result[['sequence', 'class', 'labels', 'scores']]


df, comp_name_ls = get_df()
model = get_model()

with st.sidebar:
    st.text('---[데이터 필터]---')
    year = st.sidebar.slider(
        '⁜ 연도를 선택하세요.',
        2014, 2022, (2021)
    )
    col = st.sidebar.selectbox(
        "⁜ 분석 텍스트 필드를 선택하세요.",
        ('장점', '단점', '경영진에게')
    )
    company_name = st.sidebar.selectbox(
        "⁜ 회사명을 입력/선택하세요.",
        comp_name_ls
    )
    idx = st.sidebar.text_input(
        "⁜ 조회할 데이터 시작 인덱스를 입력하세요.",
        ""
    )
    sample_n = st.sidebar.slider(
        "⁜ 조회할 데이터 총 개수를 선택하세요.",
        1, 30, (10)
    )
    user_input = st.sidebar.text_input(
        "⁜ 레이블을 입력하세요. 콤마로 분리하세요.",
        ""
    )
    multi_label_input = st.radio(
            "⁜ 멀티 레이블을 키고 끌 수 있습니다.",
            ('On', 'Off')
        )

if user_input:
    candidate_labels = [x.strip() for x in user_input.split(',')]
else:
    candidate_labels = ['복지 및 급여', '워라밸', '사내문화', '승진 기회 및 가능성']
if not idx:
    idx = 0

df_comp = get_df_by_comp(df, company_name)
df_year = get_df_by_year(df_comp, year)

col_dic = {'장점': 'Pros', '단점': 'Cons', '경영진에게': 'To_Management'}

st.title('[그레이비랩 기업부설 연구소 / AI lab.]')
st.markdown(f'레이블 : [{candidate_labels}]')
st.checkbox("넓이 자동 맞춤", value=False, key="use_container_width")
docs = df_year[col_dic[col]].apply(prep.preprocess_text).tolist()[int(idx):int(idx)+sample_n]
result = get_result(model, docs, candidate_labels, multi_label_input)
st.dataframe(result, use_container_width=st.session_state.use_container_width)
