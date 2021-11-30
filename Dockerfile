FROM nvidia/cuda:9.0-devel-ubuntu16.04

WORKDIR /app/service

ENV TZ=Europe/Kiev
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

ENV PYTHONPATH=/app/service/src
ENV FLASK_APP=src
ENV FLASK_ENV=development
ENV MONGO_HOST=host.docker.internal
ENV MONGO_PORT=27017
ENV MONGO_DB=analysis
ENV KAFKA_HOST=host.docker.internal
ENV KAFKA_PORT=9092

# Update ubuntu
RUN echo "Updating Ubuntu and installing opencv"
RUN apt-get update \
	&& apt-get install -y --no-install-recommends apt-utils \
	build-essential \
	cmake\
	git\
	libgtk2.0-dev\
	pkg-config\
	libavcodec-dev\
	libavformat-dev\
	libswscale-dev\
	libopencv-dev\
	vim \
	wget \
	curl

# Install Python
RUN echo "installing python"
RUN apt-get -y install software-properties-common
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get -y install python3.8 python3.8-distutils python3-dev liblapack3 libblas-dev liblapack-dev gfortran
RUN update-alternatives --remove python /usr/bin/python2
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 10
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python get-pip.py
RUN rm get-pip.py

# Install Python Libraries
COPY requirements.txt  /app/service
RUN echo "installing python libraries"
RUN pip3 install -r requirements.txt
COPY other_sources  /app/service

RUN nvidia-smi

CMD flask run --host=::
