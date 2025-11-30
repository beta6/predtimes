# Dockerfile
FROM python:3.9

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

COPY predtimes/requirements.txt /app/
RUN pip install -r requirements.txt

COPY . /app/
