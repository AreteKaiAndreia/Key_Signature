FROM python:3.11-slim-bookworm
WORKDIR /app
COPY requirements.txt packages.txt ./
RUN apt-get update && xargs apt-get install -y --no-install-recommends < packages.txt \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --no-cache-dir --only-binary=:all: -r requirements.txt \
    && python -m pip check
COPY .streamlit .streamlit
COPY keyclf keyclf
COPY streamlit_app.py ./
EXPOSE 8501
CMD ["python", "-m", "streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0"]
