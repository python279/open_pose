FROM python:3.8-buster

RUN echo "deb https://mirrors.aliyun.com/debian/ buster main contrib non-free" > /etc/apt/sources.list && \
    echo "deb https://mirrors.aliyun.com/debian/ buster-updates main contrib non-free" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.aliyun.com/debian/ buster-backports main contrib non-free" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.aliyun.com/debian-security/ buster/updates main contrib non-free" >> /etc/apt/sources.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends zip unzip libgl1-mesa-glx \
    && apt-get clean

WORKDIR /open_pose
COPY . /open_pose
RUN mkdir checkpoint \
    && curl -o checkpoint/models--lllyasviel--Annotators.zip "http://mirrors.uat.enflame.cc/enflame.cn/maas/open_pose/models--lllyasviel--Annotators.zip" \
    && unzip checkpoint/models--lllyasviel--Annotators.zip -d checkpoint \
    && rm checkpoint/models--lllyasviel--Annotators.zip
RUN pip3 install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip3 install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

ENTRYPOINT ["python3", "server.py"]