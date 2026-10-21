FROM python:3.12-slim

WORKDIR /usr/src/app
COPY . .
RUN apt-get update
RUN apt-get install -y libx11-xcb1 libgl1 libglib2.0-0
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8080

# Single worker: Gradio keeps its queue, session state and uploads in one process
CMD ["sh", "-c", "hypercorn app:server --bind 0.0.0.0:${PORT:-8080}"]
