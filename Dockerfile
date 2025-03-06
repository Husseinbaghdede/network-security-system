FROM python:3.10-slim-buster

WORKDIR /app
COPY . /app

RUN apt update -y && apt install -y awscli && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
