FROM ubuntu:18.04

ENV INST_PATH=/temp_inst_files
ENV PROJECT_DIR=/root/SRD

RUN  apt-get update \
  && apt-get install -y software-properties-common \
  && apt-get install -y wget curl python3-pip python-dev build-essential unzip wget\
  && apt-get update \
  && rm -rf /var/lib/apt/lists/*

WORKDIR $INST_PATH
ADD requirements.txt $INST_PATH
RUN pip3 install --trusted-host pypi.python.org -r requirements.txt

RUN apt-get update \
    && apt-get install -y apt-file \
    && apt-get -y install nano

ENV PYTHONPATH="$PYTHONPATH:/root/SRD"
#ENV PYTHONPATH="$PYTHONPATH:/root"

WORKDIR $PROJECT_DIR
COPY APP $PROJECT_DIR/APP
COPY params.py $PROJECT_DIR
COPY main $PROJECT_DIR/main

ENTRYPOINT ["python3"]
CMD ["APP/app.py"]