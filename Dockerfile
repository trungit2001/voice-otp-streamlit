FROM python:3.8

RUN apk -yq update
RUN apt-get -yq upgrade
RUN apt-get -yq autoremove
RUN apt-get -yq install gcc
RUN apt-get -yq install libasound-dev
RUN apt-get -yq install portaudio19-dev
RUN apt-get -yq install libportaudio2
RUN apt-get -yq install libportaudiocpp0
RUN apt-get -yq install ffmpeg
RUN apt-get -yq install python3-pip
RUN pip install pyaudio

WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
EXPOSE 8501
COPY . /app

RUN wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1_ngoiTYO2Fup0lPFSFK1Vl-6ggIPsLdc' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1_ngoiTYO2Fup0lPFSFK1Vl-6ggIPsLdc" -O weights.zip && rm -rf /tmp/cookies.txt
RUN mv /app/weights.zip /app/models/weights
RUN unzip -o /app/models/weights/weights.zip -d /app/models/weights

CMD streamlit run app.py --server.port $PORT