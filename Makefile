setup:
	pip install -r requirements.txt

setup-dev:
	make setup
	pip install -r requirements-dev.txt
	pre-commit install

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
