FROM amd64/ubuntu:18.04

ENV PATH /opt/conda/bin:$PATH
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-get update && apt-get upgrade -y && apt-get install -y curl git wget ffmpeg libsm6 libxext6

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

CMD [ "/bin/bash" ]

COPY ./ ./
RUN conda env create -f environment.yml

RUN conda init

SHELL [ "/bin/bash", "--login", "-c" ]
RUN conda activate dhp && conda env list && cd python && python3 driver.py