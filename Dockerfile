FROM python:3.10-slim-buster

WORKDIR /app
COPY . /app

RUN apt update -y && apt install awscli -y

RUN apt update -y && pip install -r requirements.txt

EXPOSE 8080

CMD ["python3","app.py"]