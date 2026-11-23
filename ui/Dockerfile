FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY .streamlit ./.streamlit

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

# Cloud Run injects $PORT (8080). Locally we default to 8501.
# CORS/XSRF protection is off because Cloud Run's proxy sits in front of Streamlit.
# Theme and toolbar settings live in .streamlit/config.toml.
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true --server.enableCORS=false --server.enableXsrfProtection=false"]
