FROM ubuntu:20.04

ENV TZ=Europe/Rome
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y \
  build-essential \
  software-properties-common \
  curl \
  wget \
  git

# Install Miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

# Activate environment
COPY env.yml /env.yml
RUN conda env create --file /env.yml
RUN echo "source activate env-cpu" > ~/.bashrc
ENV PATH /opt/conda/envs/env/bin:$PATH

ENV PYTHONDONTWRITEBYTECODE=1
ENV MY_ROOT_APPLICATION /custom-docker-test
WORKDIR $MY_ROOT_APPLICATION

CMD ["bash"]




