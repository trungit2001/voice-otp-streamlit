FROM python:3.8

WORKDIR /app
ADD . /app

RUN pip install -r requirements.txt

ENTRYPOINT ["streamlit", "run"]
CMD ["app.py"]
EXPOSE 8501