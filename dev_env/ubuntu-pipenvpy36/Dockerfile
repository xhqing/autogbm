FROM xhq123/ubuntu-conda

WORKDIR /app/workdir

RUN pip install pipenv
COPY pipenv_patch.py /root

CMD ["/bin/zsh"]
