setup:
	pip install -r requirements.txt

setup-dev:
	make setup
	pip install -r requirements-dev.txt

check:
	make format
	make lint

format:
	black . --line-length 110
	isort . --profile black

lint:
	flake8 . --max-line-length 110
