# Docker file for selfrep naacl experiments
FROM nvcr.io/nvidia/pytorch:23.05-py3 as base

ADD https://cmake.org/files/v3.7/cmake-3.7.2-Linux-x86_64.sh /cmake-3.7.2-Linux-x86_64.sh
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    curl \
    enchant-2 \
    llvm \
    git \
    cmake \
    neovim \
    vim \
    python3.10-venv \
    aspell-* \
    hunspell-ko \
    hunspell-ar \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/* && \
    export HOME=/home

WORKDIR /home

SHELL ["/usr/bin/bash", "-c"]

ENV HF_HOME="/home/code/hf_cache_home"
RUN mkdir /home/code/
RUN mkdir /home/code/.hf_cache_home

RUN export HOME=/home && \
    pip install --upgrade pip \
    pip install transformers==4.35.2 pyenchant && \
    pip install evaluate datasets numpy tokenizers wandb sentencepiece sacrebleu unbabel-comet==2.2.0 tensorboard==2.14 

# apply comet patch because pypy package is not updated
# file from https://github.com/ymoslem/COMET/blob/8b5103bf6baedd9c243331f82581c8f96b9b6aba/comet/models/base.py
RUN rm /usr/local/lib/python3.10/dist-packages/comet/models/base.py
ADD base.py /usr/local/lib/python3.10/dist-packages/comet/models/base.py

# fix problems with wandb autentication
RUN > /home/.netrc
# change to userid
RUN chown <uid> /home/.netrc 
# change to userid
RUN chown :<guid> /home/.netrc
RUN mkdir /home/code/space
WORKDIR /home/code/

