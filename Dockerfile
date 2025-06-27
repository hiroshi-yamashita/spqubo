# FROM python:3.11
FROM python@sha256:a3e280261e448b95d49423532ccd6e5329c39d171c10df1457891ff7c5e2301b

ENV SNAPSHOT_DATE=20250407T000000Z

RUN echo "deb http://snapshot.debian.org/archive/debian/${SNAPSHOT_DATE}/ bookworm main" > /etc/apt/sources.list && \
    echo "deb http://snapshot.debian.org/archive/debian-security/${SNAPSHOT_DATE}/ bookworm-security main" >> /etc/apt/sources.list && \
    echo "deb http://snapshot.debian.org/archive/debian/${SNAPSHOT_DATE}/ bookworm-updates main" >> /etc/apt/sources.list && \
    echo 'Acquire::Check-Valid-Until "false";' > /etc/apt/apt.conf.d/99no-check-valid-until && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    tzdata \ 
    build-essential \
    git \
    fonts-noto && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN fc-cache -fv

RUN pip install pandas==1.5.3
RUN pip install numpy==1.26.0

RUN pip install dvc==3.59.1
RUN pip install jupyter==1.1.1
RUN pip install matplotlib==3.9.0

RUN pip install cython==3.0.12
RUN pip install pulp==3.1.1

RUN pip install scipy==1.15.2

RUN mkdir /host_volume

WORKDIR /host_volume