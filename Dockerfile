FROM python:3.11-slim

WORKDIR /app

# Copiamos y ejecutamos el requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos el resto del código
COPY . .

# Exponemos el puerto que espera Hugging Face Spaces
EXPOSE 7860

# Comando para iniciar la app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]