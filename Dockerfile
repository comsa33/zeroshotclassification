FROM python:3.9.16-slim-bullseye

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUTF8=1 \
    PIP_NO_CACHE_DIR=on \
    PIP_DISABLE_PIP_VERSION_CHECK=on

WORKDIR /usr/src

RUN apt-get update -y && apt-get upgrade -y
RUN pip install pymongo pandas plotly streamlit stqdm transformers
RUN pip install transformers[sentencepiece]
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

COPY . /usr/src/ZEROSHOTCLASSIFICATION
WORKDIR /usr/src/ZEROSHOTCLASSIFICATION/

EXPOSE 8501

CMD ["streamlit", "run", "run.py"]