# start by pulling the python image
# FROM nvidia/cuda:11.8.0-devel-ubuntu18.04
FROM continuumio/miniconda3

RUN apt update
RUN apt add make automake gcc g++ subversion python3-dev
RUN apt-get update && \
    apt-get install -y wget bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x Miniconda3-latest-Linux-x86_64.sh && \
    ./Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh

# copy the requirements file into the image
COPY ./environment.yml /app/environment.yml

# copy pose related modules
# COPY ../pose/configs  /app/pose/configs
# COPY ../pose/vitpose-h-simple.pth /app/pose/vitpose-h-simple.pth

# switch working directory
WORKDIR /app

RUN conda env create -f environment.yml 

# install the dependencies and packages in the requirements file
# RUN pip install -r requirements.txt

# copy every content from the local file to the image
# COPY . /app

# configure the container to run in an executed manner
ENTRYPOINT [ "python" ]

CMD ["app.py" ]