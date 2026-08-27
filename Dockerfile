FROM python:3.12-slim

WORKDIR /usr/src/app
COPY . .
RUN apt-get update
RUN apt-get install -y libx11-xcb1 libgl1 openjdk-25-jre-headless
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

CMD ["python", "app.py"]
