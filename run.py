import pickle
from collections import Counter
from PIL import Image

from wordcloud import WordCloud
import pandas as pd
from kiwipiepy import Kiwi
from kiwipiepy.utils import Stopwords
from transformers import pipeline
import plotly.express as px
import plotly.graph_objects as go
import matplotlib as mpl
import matplotlib.pyplot as plt
import squarify
from stqdm import stqdm
import streamlit as st

import functions as funcs
import preprocess as prep
import mongodb


mpl.rcParams['axes.unicode_minus'] = False
plt.rcParams["font.family"] = 'NanumGothicCoding'

prep.make_user_dictionary()

kiwi = Kiwi()
kiwi.load_user_dictionary('user_dictionary.txt')
kiwi.prepare()
stopwords = Stopwords()

st.set_page_config(
    page_title="리뷰데이터 제로샷 자연어 추론",
    page_icon="🤖",
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
    model = pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7")
    return model

def test_sample_text(_model, sample_text, candidate_labels, multi_label_input):
    multi_label = True if multi_label_input == "ON" else False
    output = _model(sample_text, candidate_labels, multi_label=multi_label)
    return pd.DataFrame(output['scores'], output['labels'], columns=['scores'])

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
    return score_df.mean().reset_index().sort_values(by='index')

def get_all_score_dfs(df, col, _model, candidate_labels, multi_label_input, idx, sample_n):
    yealy_score_dfs = []
    all_years = df['year'].unique().tolist()
    for yr in stqdm(all_years):
        df_year_ = get_df_by_year(df, yr)
        docs_by_year = df_year_[col].apply(prep.preprocess_text).tolist()
        result_by_year = get_result(_model, docs_by_year, candidate_labels, multi_label_input, idx, sample_n)
        yealy_score_dfs.append(get_score_avg_by_label(result_by_year))
    return yealy_score_dfs, all_years

@st.experimental_singleton
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

@st.experimental_singleton
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

st.session_state.df, st.session_state.comp_name_ls = get_df()
st.session_state.model = get_model()

with st.sidebar:
    st.text('[데이터 필터]')
    year = st.slider(
        '1️⃣ 연도를 선택하세요.',
        2014, 2022, (2022)
    )
    col = st.selectbox(
        "2️⃣ 분석 텍스트 필드를 선택하세요.",
        ('장점', '단점', '경영진에게')
    )
    company_name = st.selectbox(
        "3️⃣ 회사명을 입력/선택하세요.",
        st.session_state.comp_name_ls
    )

col_dic = {'장점': 'Pros', '단점': 'Cons', '경영진에게': 'To_Management'}

df_company = get_df_by_comp(st.session_state.df, company_name)
df_year = get_df_by_year(df_company, year)
n_df_year = len(df_year) 
n_df_year_limit = n_df_year if n_df_year < 101 else 100

st.title('[그레이비랩 기업부설 연구소 / AI lab.]')

with st.container():
    default_candidate_labels = ['복지 워라밸', '연봉 돈 급여', '수직 꼰대', '도전 성장 가능성', '수평 자율', '동료 협업', '야근 일']
    user_input = st.text_input(
    f"✓ 사용자 레이블을 입력하시고, 콤마로 분리하세요.\n\t(default={default_candidate_labels})",
    ""
    )
    if user_input:
        candidate_labels = [x.strip() for x in user_input.split(',')]
    else:
        candidate_labels = default_candidate_labels

    col1, _, col2, _, col3 = st.columns([5,1,5,1,5])
    with col1:
        idx = st.text_input(
            "✓ 조회할 데이터 시작 인덱스를 입력하세요. (defalut=0)",
            ""
        )
    with col2:
        st.checkbox(f"전체 데이터 선택 (전체 데이터 개수:{n_df_year}, 100개 이상의 경우 100으로 제한)", value=False, key="use_all_yealy_data")
        if st.session_state.use_all_yealy_data:
            sample_n = n_df_year_limit
        else:
            sample_n = st.slider(
                "✓ 딥러닝 모델에 추론할 데이터 총 개수를 선택하세요.",
                1, 30, (10)
            )
    with col3:
        multi_label_input = st.radio(
            "✓ 멀티 레이블을 키고 끌 수 있습니다.",
            ('On', 'Off')
        )

    if not idx:
        idx = 0

tab1, tab2, tab3, tab4 = st.tabs(["✅ 샘플 텍스트 테스트", "🗃 리뷰 데이터 테스트", "📈 연도별 트렌드 결과 비교", "🏷️ 레이블 키워드 관련 빈출 어휘"])

with tab1:
    with st.expander("❓ 자세히 보기 : 사용한 DL model - [mDeBERTa-v3-base-xnli-multilingual-nli-2mil7]"):
        st.markdown(
            """
- 이 다국어 모델은 100개 언어에 대해 자연어 추론(NLI)을 수행할 수 있으므로 다국어 제로샷 분류에도 적합합니다. 기본 mDeBERTa-v3-base 모델은 100개 언어로 구성된 CC100 다국어 데이터 세트에서 Microsoft에 의해 사전 훈련되었습니다. 그런 다음 모델은 XNLI 데이터 세트와 다국어 NLI-26lang-2mil7 데이터 세트에서 fine-tune되었습니다. 두 데이터 세트 모두 40억 명이 넘는 사람들이 사용하는 27개 언어로 된 270만 개 이상의 가설-전제 쌍을 포함합니다.
            """
        )
    tab1_col1, _, tab1_col2 = st.columns([4,1,2])
    with tab1_col1:
        sample_text = st.text_area(
            "✓ 분류하고자 하는 샘플 텍스트를 입력하세요.",
            """연 베네핏카드 300만원과 휴양시설 이용비 지원 그리고 3년 재직자에게 1개월의 리프레쉬 휴가 등 여러 복지가 잘 되어 있으며 신규 프로젝트 진행시 대부분의 경험상 바텀업 형식으로 일정 조율이 되어 야근할일이 거의 없음. 출산휴가 육아휴직 매우 자유롭고 눈치 안보이고 특히 자녀가 있을 경우 사내어린이집이 매우 유용함. 업무 진행시 동료들 성격이 대부분 둥글둥글해서 사람 스트레스가 거의 없는 편이다"""
        )
    with tab1_col2:
        if sample_text:
            sample_result = test_sample_text(st.session_state.model, sample_text, candidate_labels, multi_label_input)
            st.dataframe(sample_result)

with tab2:
    st.subheader(f'{year}년 {company_name}-{col} 샘플 결과')

    tab2_col1, tab2_col2 = st.columns([2, 1])

    with tab2_col1:
        docs_sample = df_year[col_dic[col]].apply(prep.preprocess_text).tolist()
        result = get_result(st.session_state.model, docs_sample, candidate_labels, multi_label_input, idx, sample_n)
        st.dataframe(result)
        st.caption(f"{year}년 {company_name}추론 결과표")

    with tab2_col2:
        score_avg = get_score_avg_by_label(result)
        draw_radar_chart(score_avg)
        st.caption(f"{year}년 {company_name} 각 레이블 평균 추론 스코어")

with tab3:
    st.subheader(f'{company_name}-{col} 연도별 트렌드 결과')

    yealy_score_dfs, all_years = get_all_score_dfs(
        df_company, col_dic[col], st.session_state.model, candidate_labels, multi_label_input, idx, sample_n
    )
    draw_radar_charts_yearly(yealy_score_dfs, all_years)

with tab4:
    st.subheader(f'{year}년 {company_name}-{col} 레이블별 관련 빈출 어휘 그래프')
    tab4_col1, _, tab4_col2, _, tab4_col3 = st.columns([5,1,5,1,5])
    with tab4_col1:
        label_selected = st.selectbox(
            "✓ 레이블 명을 입력/선택하세요.",
            candidate_labels
        )
    with tab4_col2:
        n_words = st.slider(
            "✓ 그래프에서 보여줄 단어의 수를 선택하세요.",
            20, 50, (30)
        )
    with tab4_col3:
        style = st.radio(
            "✓ 시각화 스타일을 선택할 수 있습니다.",
            ('squarify', 'wordcloud')
        )
    draw_word_plot(result, label_selected, n_words, style=style)
