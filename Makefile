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
	@files="requirements.in requirements-dev.in"; \
	for file in $$files; do \
		if [[ -f $$file ]]; then \
			sort -f -o $$file $$file; \
		fi \
	done

check:
	make format
	make lint

format:
	black . --line-length 110
	isort . --profile black

lint:
	flake8 . --max-line-length 110 --extend-ignore E203

up:
	docker compose up -d

down:
	docker compose down -v

rmi:
	docker rmi k8s-docs-assistant-rag-vector-db k8s-docs-assistant-rag-api k8s-docs-assistant-rag-interface

restart:
	make down
	make rmi
	make up
