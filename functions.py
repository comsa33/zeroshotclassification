from collections import Counter
import pickle

import pandas as pd
from sqlalchemy import text
from kiwipiepy import Kiwi
from kiwipiepy.utils import Stopwords
from wordcloud import WordCloud
from transformers import pipeline
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import squarify
from stqdm import stqdm
import streamlit as st

import settings
import queries as nq
import preprocess as prep
from postgre import postgre_engine as engine


mpl.rcParams['axes.unicode_minus'] = False
plt.rcParams["font.family"] = 'NanumGothicCoding'

kiwi = Kiwi()
kiwi.load_user_dictionary('user_dictionary.txt')
kiwi.prepare()
stopwords = Stopwords()

filename = ['jp_comp_name_list']
comp_name_ls = tuple(pickle.load(open(filename[0], 'rb')))


@st.cache_data
def get_data():
    with engine.connect() as conn:
        fetch = conn.execute(text(nq.FindAllFromJobplanetReview)).fetchall()

    jp_df = pd.DataFrame(fetch)
    jp_df['all_text'] = jp_df['pros']+jp_df['cons']+jp_df['to_managements']
    return jp_df


def get_comp(df, company_name):
    df_ = df[df['Company_name'] == company_name]
    df_['DatePost'] = pd.to_datetime(df_['DatePost'], errors='coerce')
    df_['year'] = df_['DatePost'].apply(lambda x: x.year)
    return df_


@st.cache_data
def get_df_by_comp(df, company_name):
    df_comp = get_comp(df, company_name)
    return df_comp


@st.cache_data
def get_df_by_year(df, year):
    df_year = df.query(f'year == {year}')
    return df_year


@st.cache_data
def get_model():
    model = pipeline("zero-shot-classification", model=settings.model_name)
    return model


def test_sample_text(_model, sample_text, candidate_labels, multi_label_input):
    multi_label = True if multi_label_input == "ON" else False
    output = _model(sample_text, candidate_labels, multi_label=multi_label)
    try:
        output['labels'] = label_mapping(output['labels'])
    except:
        pass
    return pd.DataFrame(output['scores'], output['labels'], columns=['scores'])


@st.cache_data
def get_result(_model, docs, candidate_labels, multi_label_input, idx, sample_n):
    multi_label = True if multi_label_input == "ON" else False
    outputs = []
    for doc in stqdm(docs[int(idx):int(idx)+sample_n]):
        output = _model(doc, candidate_labels, multi_label=multi_label)
        outputs.append(output)
    result = pd.DataFrame(outputs)
    try:
        result['labels'] = result['labels'].apply(label_mapping)
    except:
        pass
    result['class'] = result['labels'].apply(lambda x: x[0])
    return result[['sequence', 'class', 'labels', 'scores']]


def label_mapping(labels):
    new_labels = []
    for label in labels:
        new_labels.append(label_dict_selected[label])
    return new_labels


@st.cache_data
def get_score_avg_by_label(result):
    dicts = []
    for labels, scores in list(zip(result['labels'].tolist(), result['scores'].tolist())):
        dicts.append(dict(zip(labels, scores)))
    score_df = pd.DataFrame(dicts)
    return score_df.mean().reset_index().sort_values(by='index')


@st.cache_data
def get_all_score_dfs(df, col, _model, candidate_labels, multi_label_input, idx, sample_n):
    yealy_score_dfs = []
    all_years = sorted(df['year'].unique().tolist())
    for yr in stqdm(all_years):
        df_year_ = get_df_by_year(df, yr)
        docs_by_year = df_year_[col].apply(prep.preprocess_text).tolist()
        result_by_year = get_result(_model, docs_by_year, candidate_labels, multi_label_input, idx, sample_n)
        yealy_score_dfs.append(get_score_avg_by_label(result_by_year))
    return yealy_score_dfs, all_years


@st.cache_resource
def draw_radar_chart(df):
    fig = px.line_polar(df, r=0, theta='index', line_close=True)
    fig.update_traces(fill='toself')
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            ))
    )
    st.plotly_chart(fig, use_container_width=True)


@st.cache_resource
def draw_radar_charts_yearly(dfs, all_years):
    fig = go.Figure()
    for year, df in zip(all_years, dfs):
        fig.add_trace(
            go.Scatterpolar(
                r=df[0].tolist(),
                theta=df['index'].tolist(),
                fill='toself',
                name=f'{year}'
            )
        )
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        )
    )
    st.plotly_chart(fig, use_container_width=True)


@st.cache_resource
def draw_word_plot(result, label_selected, n_words, style='squarify'):
    sents_by_class = ' '.join(result[result['class']==f"{label_selected}"]['sequence'].tolist())

    tokens = stopwords.filter(kiwi.tokenize(sents_by_class))
    nouns = []
    for token in tokens:
        if token.tag in ['NNG', 'NNP', 'SL']:
            nouns.append(token.form)
    
    if nouns:
        cnt_nouns = Counter(nouns).most_common(n_words)
        nouns_df = pd.DataFrame(cnt_nouns, columns=['words', 'count'])

        if style == 'squarify':
            fig = plt.figure(figsize=(10, 5))
            squarify.plot(nouns_df['count'], label = nouns_df['words'], color=plt.cool(), alpha=0.5, edgecolor="white", linewidth=2)
            plt.axis('off')

        elif style == 'wordcloud':
            word_cloud = WordCloud(font_path='/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
                            width = 1000, height = 500,
                            background_color='white')
            word_cloud.generate_from_frequencies(dict(cnt_nouns))
            fig = plt.figure(figsize=(10, 5))
            plt.imshow(word_cloud)
            plt.axis("off")
            plt.tight_layout(pad=0)
        st.pyplot(fig)
    else:
        st.write("표시할 명사형 어휘가 존재하지 않습니다.")
