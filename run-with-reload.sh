# remember, in prod: Gunicorn with Uvicorn workers
python -m uvicorn server:app --reload --host 0.0.0.0 --port 9000
