# imagen base: python
FROM python:3.10-slim

ENV PORT 8000
# Instala dependencias de OpenCV en la imagen
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
# Copia archivo de requerimientos
COPY requirements.txt /
# Instala dependencias con pip
RUN pip install -r requirements.txt
# Copia el codigo fuente y archivos necesarios
COPY . /app
WORKDIR /app

CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT}