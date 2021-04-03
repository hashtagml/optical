# [Choice] Python version: 3, 3.9, 3.8, 3.7, 3.6
ARG VARIANT=3.8
FROM mcr.microsoft.com/vscode/devcontainers/python:${VARIANT}

RUN apt-get update && apt-get install -y curl

ENV POETRY_HOME="/opt/poetry"
ENV PATH="$POETRY_HOME/bin:$PATH"

WORKDIR /workspaces/optical/
COPY poetry.lock pyproject.toml /workspaces/optical/

RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
RUN poetry config virtualenvs.create false
RUN poetry install

COPY . /workspaces/optical/
