# 使⽤官⽅ Python 轻量级镜像 
# https://hub.docker.com/_/python 
FROM python:3.6-slim 
# 将本地代码拷贝到容器内 
ENV APP_HOME /novel 
WORKDIR /novel 
ADD . /novel 
# 安装依赖 
RUN pip3 install Flask gunicorn 
RUN pip3 install opencv-python
RUN pip3 install mediapipe
RUN pip3 install bumpy
RUN pip3 install matplotlib
EXPOSE 5000 
# 启动 Web 服务 
# 这⾥我们使⽤了 gunicorn 作为 Server，1 个 worker 和 8 个线程 
# 如果您的容器实例拥有多个 CPU 核⼼，我们推荐您把线程数设置为与 CPU 核⼼数⼀致 
CMD ["python", "骨架.py"]