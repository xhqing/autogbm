FROM xhq123/ubuntu-conda

WORKDIR /app/workdir
RUN mkdir autogbm

RUN apt install -y vim 
# RUN apt install -y gcc
RUN apt install -y git

RUN apt install -y zsh
COPY ohmyzsh_install.sh .
RUN sh ohmyzsh_install.sh
COPY .zshrc /root
RUN git clone https://github.com/xhqing/zsh-autosuggestions.git ~/.zsh/zsh-autosuggestions

RUN apt install -y pip && pip install pipenv

CMD ["/bin/zsh"]
