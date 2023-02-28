import streamlit as st

import functions as funcs
import preprocess as prep


st.set_page_config(
    page_title="리뷰데이터 제로샷 자연어 추론",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="auto",
)
prep.make_user_dictionary()
filename = ['jp_comp_name_list']
st.session_state.comp_name_ls = funcs.comp_name_ls


if st.button('GET DATA'):
    st.session_state.df = funcs.get_data()

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

try:
    col_dic = {'장점': 'pros', '단점': 'cons', '경영진에게': 'to_management'}

    df_company = funcs.get_df_by_comp(st.session_state.df, company_name)
    df_year = funcs.get_df_by_year(df_company, year)
    n_df_year = len(df_year) 
    n_df_year_limit = n_df_year if n_df_year < 101 else 100

    st.title('[그레이비랩 기업부설 연구소 / AI lab.]')

    st.session_state.label_dict = {
        '장점': {
                    "도전": "개척, 변화, 새로운시도, 성장",
                    "창의성": "아이디어, 유연한사고, 독창성, 상상",
                    "소통/협력": "수평, 팀웍, 동료, 협업",
                    "원칙": "도덕성, 정직, 무결점, 공정, 기본, 정도",
                    "책임감": "도덕성, 정직, 공정 , 문서"
        },
        '단점': {
                    "도전": "개척, 변화, 새로운시도, 성장",
                    "창의성": "아이디어, 유연한사고, 독창성, 상상",
                    "소통/협력": "수평, 팀웍, 동료, 협업",
                    "원칙": "도덕성, 정직, 무결점, 공정, 기본, 정도",
                    "책임감": "도덕성, 정직, 공정 , 문서"
        }
    }

    st.session_state.model = funcs.get_model()
    label_dict_selected = dict([(value, key) for key, value in st.session_state.label_dict[col].items()])

    with st.container():
        default_candidate_labels = ["도전", "창의성", "소통/협력", "원칙", "책임감"]
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
                sample_result = funcs.test_sample_text(st.session_state.model, sample_text, candidate_labels, multi_label_input, label_dict_selected)
                st.dataframe(sample_result)

    with tab2:
        st.subheader(f'{year}년 {company_name}-{col} 샘플 결과')

        tab2_col1, tab2_col2 = st.columns([2, 1])

        with tab2_col1:
            docs_sample = df_year[col_dic[col]].apply(prep.preprocess_text).tolist()
            result = funcs.get_result(st.session_state.model, docs_sample, candidate_labels, multi_label_input, idx, sample_n)
            st.dataframe(result)
            st.caption(f"{year}년 {company_name}추론 결과표")

        with tab2_col2:
            score_avg = funcs.get_score_avg_by_label(result)
            funcs.draw_radar_chart(score_avg)
            st.caption(f"{year}년 {company_name} 각 레이블 평균 추론 스코어")

    with tab3:
        st.subheader(f'{company_name}-{col} 연도별 트렌드 결과')

        yealy_score_dfs, all_years = funcs.get_all_score_dfs(
            df_company, col_dic[col], st.session_state.model, candidate_labels, multi_label_input, idx, sample_n
        )
        funcs.draw_radar_charts_yearly(yealy_score_dfs, all_years)

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
        funcs.draw_word_plot(result, label_selected, n_words, style=style)

except AttributeError:
    st.write('먼저 "GET DATA" 버튼을 눌러 데이터를 불러오세요.')
