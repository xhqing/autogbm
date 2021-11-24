FROM xhq123/ubuntu-zsh

WORKDIR /app/workdir

COPY conda_install.sh .

RUN apt install -y vim
RUN apt install -y pip

CMD ["/bin/zsh"]
