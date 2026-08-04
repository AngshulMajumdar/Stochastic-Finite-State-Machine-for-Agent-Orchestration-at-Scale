FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

WORKDIR /workspace

COPY requirements.txt pyproject.toml README.md ./
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY src ./src
RUN python -m pip install --no-cache-dir -e . --no-build-isolation --no-deps

COPY experiments ./experiments
COPY results/reference ./results/reference
COPY tests ./tests

CMD ["python", "experiments/run_all.py", "--profile", "quick"]
