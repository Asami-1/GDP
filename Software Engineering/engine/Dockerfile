# start by pulling the python image
FROM nvidia/cuda:11.7.0-devel-ubuntu18.04
RUN apk update
RUN apk add make automake gcc g++ subversion python3-dev

# copy the requirements file into the image
COPY ./requirements.txt /app/requirements.txt

# copy pose related modules
COPY ../pose/configs  /app/pose/configs
COPY ../pose/vitpose-h-simple.pth /app/pose/vitpose-h-simple.pth

# switch working directory
WORKDIR /app

RUN pip install --upgrade pip setuptools wheel

# install the dependencies and packages in the requirements file
RUN pip install -r requirements.txt

# copy every content from the local file to the image
COPY . /app

# configure the container to run in an executed manner
ENTRYPOINT [ "python" ]

CMD ["app.py" ]