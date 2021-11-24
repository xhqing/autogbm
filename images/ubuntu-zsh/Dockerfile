FROM ubuntu

WORKDIR /app/workdir
 
RUN apt update && apt install -y zsh

COPY ohmyzsh_install.sh .
RUN apt install -y git
RUN sh ohmyzsh_install.sh && rm ohmyzsh_install.sh

RUN git clone https://github.com/xhqing/zsh-autosuggestions.git ~/.zsh/zsh-autosuggestions
COPY .zshrc /root

CMD ["/bin/zsh"]
