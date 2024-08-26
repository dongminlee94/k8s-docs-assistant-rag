setup:
	pip install -r requirements.txt

setup-dev:
	make setup
	pip install -r requirements-dev.txt
	pre-commit install

requirements:
	pip-compile requirements.in
	pip-compile requirements-dev.in

requirements-in-fixer:
	python -c "with open('requirements.in', 'r') as f: lines = sorted(f.readlines(), key=str.lower); open('requirements.in', 'w').writelines(lines)"
	python -c "with open('requirements-dev.in', 'r') as f: lines = sorted(f.readlines(), key=str.lower); open('requirements-dev.in', 'w').writelines(lines)"

check:
	make format
	make lint

format:
	black . --line-length 110
	isort . --profile black

lint:
	flake8 . --max-line-length 110 --extend-ignore E203

chat:
	docker compose up -d

chat-end:
	docker compose down -v

rmi:
	docker rmi k8s-docs-assistant-rag-vector-db k8s-docs-assistant-rag-api k8s-docs-assistant-rag-interface

restart:
	make chat-end
	make rmi
	make chat
