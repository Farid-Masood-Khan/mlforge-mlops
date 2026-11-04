.PHONY: setup lint format typecheck test data train serve clean reproduce

PYTHON ?= python

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

lint:
	ruff check src tests

format:
	ruff format --check src tests

typecheck:
	mypy src

test:
	pytest tests/

data:
	$(PYTHON) -m src.data.download
	$(PYTHON) -m src.data.validate

train:
	$(PYTHON) -m src.models.train

promote:
	$(PYTHON) -m src.models.promote

serve:
	uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --reload

drift:
	$(PYTHON) -m src.monitoring.simulator --count 500 --perturb monthly_charges
	$(PYTHON) -m src.monitoring.report

demo:
	streamlit run demo/app.py

reproduce:
	$(PYTHON) -m src.reproduce

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} +

