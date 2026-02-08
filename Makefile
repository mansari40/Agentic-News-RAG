install:
	uv pip install -e ".[dev]"
	

check:
	pre-commit run -a

test:
	pytest


