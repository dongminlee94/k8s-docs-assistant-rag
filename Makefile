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
	@for file in requirements.in requirements-dev.in; do \
		if [ -f "$$file" ]; then \
			echo "Sorting $$file"; \
			/usr/bin/sort -f -o "$$file" "$$file"; \
		else \
			echo "$$file does not exist."; \
		fi; \
	done

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
	make down
	make rmi
	make up
