FROM python:3.6

RUN mkdir /app
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple django==1.11 djangorestframework pymysql django-redis pyjwt uwsgi requests

COPY . /app/
EXPOSE 8080

CMD ["uwsgi", "--ini", "/app/uwsgi.conf"]