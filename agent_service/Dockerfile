# FROM neuralnetx.azurecr.io/python-graphviz:3.12-slim
FROM agent-sevice-env:base

RUN apt update && apt install -y ca-certificates
RUN update-ca-certificates

WORKDIR /app
COPY ./agent_service/requirements.txt .
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY ./agent_service ./agent_service
COPY ./shared ./shared
COPY ./.env .

CMD ["uvicorn", "agent_service.app:app", "--host", "0.0.0.0", "--port", "8001"]