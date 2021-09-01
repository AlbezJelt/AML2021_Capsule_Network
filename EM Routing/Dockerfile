FROM nvcr.io/nvidia/tensorflow:21.07-tf1-py3

RUN apt-get update
RUN apt-get install wget -y

# Reconfiguring locales
# See https://serverfault.com/questions/54591/how-to-install-change-locale-on-debian
# RUN apt-get install -y locales
# RUN locale-gen en_US.UTF-8
# RUN echo "export LC_ALL=en_US.UTF-8" >> ~/.bashrc
# RUN echo "export LANG=en_US.UTF-8" >> ~/.bashrc
# RUN echo "export LANGUAGE=en_US.UTF-8" >> ~/.bashrc

RUN pip install --upgrade pip

COPY requirements.txt requirements.txt

RUN --mount=type=cache,target=/root/.cache \
    pip install -r requirements.txt

EXPOSE 5000