# start by pulling the python image
FROM python:3.8-alpine
RUN apk update
RUN apk add make automake gcc g++ subversion python3-dev

# copy the requirements file into the image
COPY ./requirements.txt /app/requirements.txt

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