FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

RUN apt update
RUN apt remove -y python3-blinker
RUN apt install -y git-lfs
RUN git lfs install --skip-repo

ADD rachleona-noize/ /noize/rachleona-noize
ADD pyproject.toml /noize/pyproject.toml

RUN cd noize; pip install -e .
EXPOSE 8888
CMD ['noize', 'web', '--port', '8888']
