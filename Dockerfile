
FROM frolvlad/alpine-python-machinelearning:latest
RUN pip3 install --upgrade pip

WORKDIR /app

COPY . /app

RUN apk add --no-cache --virtual .build-deps gcc musl-dev python3-dev\
  && pip3 install cython \
  && pip3 install --no-cache-dir -r requirements.txt \
  && apk del .build-deps gcc musl-dev

RUN python3 -m nltk.downloader punkt

EXPOSE 4000

# ENTRYPOINT ['python']

# CMD ['app.py']

CMD python /app/app.py
