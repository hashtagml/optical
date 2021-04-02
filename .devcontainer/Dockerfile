# [Choice] Python version: 3, 3.9, 3.8, 3.7, 3.6
ARG VARIANT=3.8
FROM mcr.microsoft.com/vscode/devcontainers/python:${VARIANT}


WORKDIR /project
COPY poetry.lock pyproject.toml /project/


RUN pip install poetry
RUN poetry install --no-interaction

COPY . /project
