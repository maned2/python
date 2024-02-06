FROM python:3.12.1
# RUN apk update \
#   && apk add \
#     build-base \
#     postgresql \
#     postgresql-dev \
#     libpq
RUN mkdir /usr/src/app
RUN pip install --upgrade pip
WORKDIR /usr/src/app
COPY ./requirements.txt .
RUN pip install -r requirements.txt
ENV PYTHONUNBUFFERED 1
COPY . .