.PHONY: lint format test docs clean

lint:
	ruff check .

format:
	black . && ruff format .

test:
	pytest

docs:
	mkdocs serve

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.py[co]" -delete