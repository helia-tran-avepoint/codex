# FROM neuralnetx.azurecr.io/python-graphviz:3.12-slim
FROM index-sevice-env:base

WORKDIR /app
COPY index_service/requirements.txt .
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY ./index_service ./index_service
COPY ./shared ./shared
COPY ./.env  .
RUN mkdir transform_cache && chmod 777 transform_cache
COPY ./default.sqlite .

CMD ["uvicorn", "index_service.app:app", "--host", "0.0.0.0", "--port", "8002", "--loop", "asyncio"]
