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
    try:
        output['labels'] = label_mapping(output['labels'])
    except:
        pass
    return pd.DataFrame(output['scores'], output['labels'], columns=['scores'])

@st.experimental_memo
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

@st.experimental_memo
def get_score_avg_by_label(result):
    dicts = []
    for labels, scores in list(zip(result['labels'].tolist(), result['scores'].tolist())):
        dicts.append(dict(zip(labels, scores)))
    score_df = pd.DataFrame(dicts)
    return score_df.mean().reset_index().sort_values(by='index')

@st.experimental_memo
def get_all_score_dfs(df, col, _model, candidate_labels, multi_label_input, idx, sample_n):
    yealy_score_dfs = []
    all_years = sorted(df['year'].unique().tolist())
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

@st.experimental_singleton
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

st.session_state.label_dict = {
    '장점': {
                '배려': "열린마음, 공평, 동료, 사람, 소통(커뮤니케이션, 의사소통), 존중( 선한 사람들), 평등(남녀평등, 공채출신과 경력직 출신 공정평가, 계약직과 정규직 ),  수평조직, 호칭(님문화, 영어이름), 자율성(자율복장, 근무유연제, 유연근무제), 상호작용, 워라밸(일과삶,삶과일,균형,밸런스, 육아휴직, 경조사비,재택근무, 하이브리드근무, 근무유연제,유연근무제, ), 인맥, 관용, 신뢰, 매너(에티켓,예의), 인간관계, 평온, 관심, 의리, 휴식, 구성원, 공감대 형성,  편의, 개인시간, 서글서글, 도덕성",
                '목표': "원대한, 비전, 새로운 시도, 진취,  정보 공유, 동기부여, 프로젝트, 팀, 규모 성장, 이상주의, 방향성, 도전, 최초, 최고, 목표의식",
                '학습': "미래, 신기술, 세상을 바꾼다, 성장, 연구, 최신(트렌드, 동향에 민감), 선도, 직무이동(사내이동 기회), 배움, 자기계발비, 외국어, 해외, 연수, 자발적, 글로벌, 실험적, 진출, 개척, 발전 가능성, 혁신, 개혁, 진보, 창의, 수용, 시장흐름에 민감, 스마트, 다양한 경험, 투자, 개발, 의욕, 열정, 변화",
                '즐거움': "즐겨라, 돈벌이 수단 이상, 자유(얽매이지않음), 유연(유연한 사고), 이벤트(행사,활동), 활기, 빠른소통, 애사심, 시너지(협업,화합,케미,콜라보), 만족도, 애착, 다양성(근무의 다양성), 젊은 연령대, 긍정적, 적극적, 아이디어, 특색(개성)",
                '결과': "전략, 집중, 속도, 목표 수립, 성과(성과금,금전적 보상,) 결과, 성취감, 기회, 야근비, 실적, 실력, 인정, 매출, 스톡옵션, 얼라인,  로켓성장, 승진기회, 우상향, 인센티브, 빠른실행, 불필요한 과정 생략, 보고 간소화",
                '권위': "열망, 용기, 투지, 자부심, 대외적 이미지, 인지도, 조직화, 네임 밸류(타이틀,이름,이름값, 명성, 대기업, 큰규모), 커리어, 경력직, 충성심, 경쟁력, 대우, 고연봉(연봉테이블높음), 평판, 업무 스케일, 명실상부, 1티어, 자신감, 사회적 인식, 프라이드, 고스펙",
                '안전': "자기 방어(방어적,), 전문가, 대비, 교육, 보고, 안정성(안정적),  고용유지, 신중함, 정년보장, 연금, 미래보장, 상황파악, 이해, 전망, 높은 수명, 근속 년수, 버티기 좋음, 온실 속 화초, 망할 일 없음, ",
                '질서': "규칙(룰,틀), 관리, 문서화, 체계(체계화, ), 지표, 프로그램,  리스트, 프로세스, 툴, 형식, 메뉴얼, R&R, 정교함, 짜임새, 역할, 전문성, 제도화 정확성, 확실성, 단순, 확실한 업무분담"
    },
    '단점': {
                '배려': "비난, 간섭, 성장 제한, 낮은 연봉, 낮은 급여, 의지부족, 책임감, 부담감, 잦은 회식, 잦은 행사, 조심",
                '목표': "조직 개편, 노하우 부족, 조직구조 변경(부서변경, 조직개편), 시행착오",
                '학습': "진입장벽, 비전공, 불안정, 과제, 비정규성(비정규직,계약직), 사수없음, 작은규모, 독학 필요, 평생 공부, 성장통, 실패",
                '즐거움': "대충대충, 실력없음, 주먹구구, 혼란, 혼선, 대처 미흡, 애매모호, 중구난방, 우왕좌왕, 감정적, 집중력 부족, 업무 프로세스 없음, 규율 부재, 스스로 만들어야함, 시스템, 흐지부지, 무능, 무능력, 체계 없음",
                '결과': "급하다, 빠르다,  성과측정, 경쟁, 개인주의, 불안감, 무력감, 풀타임, 압박감, 교대 근무, 쥐어짜기, 직무전환잦음, 물경력, 좌천, 저성과자 관리, 결과주의, 성과주의, 잦은 면담, 번아웃(과열, 과다), 현타, 독종, 도덕적해이, 업무강도(업무강도, 업무량, 늦은 퇴근, 야근, 주말근무, 야간근무,오버타임)",
                '권위': "연고, 상명하복, 윗사람,  승진불가능, 월급루팡, 관료주의, 꼰대, 눈치, 비합리, 비효율, 텃세, 정치( 편 가르기), 통보(가스라이팅, 명령), 박탈감, 조직문화 (군대문화), 무관심, 옛날생각, 갈아치우기(땜빵), 오너회사, 헌신 강요, 그들만의 리그, 잡일, 가족경영(학연,지연,혈연,낙하산), 권위주의자, 희생정신, 권력, 악습, 피라미드, 파벌, 갑질, 라인",
                '안전': "고인물, 공무원, 통보, 부정, 의견 묵살, 정체, 구시대(고착화, 과거방식에 매몰,뒤쳐짐, 변화 거부,) 격지발령, 우물안 개구리, 불필요 작업 많음(문서작업), 보여주기식, 보수적, 보고를 위한 보고, 나태, 매일 보고, 합리화, 방어적, 낮은 투자, 변화 지양, 계산적, 도태, 고리타분, 결제",
                '질서': "경직, 수직(딱딱함, 빡빡함), 삭막함, 융통성 부족, 반복성, 부품, 당연시,  감시자, 평가, 규제, 버튼맨, 매너리즘"
    }
}

label_dict_selected = dict([(value, key) for key, value in st.session_state.label_dict[col].items()])

with st.container():
    default_candidate_labels = ['배려', '목표', '학습', '즐거움', '결과', '권위', '안전', '질서']
    user_input = st.text_input(
    f"✓ 사용자 레이블을 입력하시고, 콤마로 분리하세요.\n\t(default={default_candidate_labels})",
    ""
    )
    if user_input:
        candidate_labels = [x.strip() for x in user_input.split(',')]
    else:
        if col in ['장점', '단점']:
            candidate_labels = [st.session_state.label_dict[col][label] for label in default_candidate_labels]
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
            [label_dict_selected[label] if label_dict_selected.get(label) else label for label in candidate_labels]
        )
    with tab4_col2:
        n_words = st.slider(
            "✓ 그래프에서 보여줄 단어의 수를 선택하세요.",
            20, 50, (30)
        )
    with tab4_col3:
        style = st.radio(
            "✓ 시각화 스타일을 선택할 수 있습니다.",
            ('wordcloud', 'squarify')
        )
    draw_word_plot(result, label_selected, n_words, style=style)
