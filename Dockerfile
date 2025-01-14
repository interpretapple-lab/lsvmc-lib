FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LIBGL_ALWAYS_INDIRECT=1

RUN apt-get update && \
    apt-get install -y -q  \ 
    apt-utils \
    python3-pip python3-venv \
    python3-numpy python3-sklearn python3-nltk python3-matplotlib python3-ete3 \
    texlive-latex-base texlive-latex-recommended texlive-latex-extra cm-super dvipng

RUN python3 -m venv --system-site-packages /app/.venv

ENV PATH="/app/.venv/bin:$PATH"

RUN python3 -m pip install --no-cache-dir \
        keras \
        tensorflow \
        jupyter 
      
WORKDIR /app
COPY . /app

EXPOSE 8888

ENV QT_QPA_PLATFORM=offscreen
ENV XDG_RUNTIME_DIR=/tmp/runtimee-root
ENV NLTK_DATA=/app/nltk_data

CMD ["jupyter", "lab", "--ip='*'", "--port=8888", "--no-browser", "--allow-root"]