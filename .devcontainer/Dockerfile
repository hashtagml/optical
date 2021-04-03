# [Choice] Python version: 3, 3.9, 3.8, 3.7, 3.6
ARG VARIANT=3.8
FROM mcr.microsoft.com/vscode/devcontainers/python:${VARIANT}


WORKDIR /workspaces/optical/
COPY poetry.lock pyproject.toml /workspaces/optical/

RUN pip install poetry
RUN poetry config virtualenvs.create false
RUN poetry install

COPY . /workspaces/optical/
