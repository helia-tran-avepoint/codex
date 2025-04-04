# FROM neuralnetx.azurecr.io/python:3.12-slim
FROM web-sevice-env:base

WORKDIR /app

COPY webui/requirements.txt .
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY ./webui ./webui
COPY ./shared ./shared
COPY ./.env ./.env
COPY ./favicon.ico ./favicon.ico

RUN rm -rf requirements.txt webui/requirements.txt

ENV PYTHONPATH=.
CMD ["streamlit", "run", "webui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
