.PHONY: install lint format test refactor all clean run-api run-cli help

help:
	@echo "Available commands:"
	@echo "  make install   - Install all dependencies"
	@echo "  make lint      - Run pylint on all Python files"
	@echo "  make format    - Format code with black"
	@echo "  make test      - Run tests with pytest and coverage"
	@echo "  make refactor  - Run format and lint"
	@echo "  make all       - Run install, format, lint, and test"
	@echo "  make run-api   - Run the FastAPI application"
	@echo "  make run-cli   - Show CLI help"
	@echo "  make clean     - Remove cache and temporary files"

install:
	@echo "Installing dependencies..."
	uv sync
	uv add pillow click fastapi uvicorn jinja2 httpx pytest pytest-cov pylint black

lint:
	@echo "Running pylint..."
	uv run python -m pylint logic/*.py cli/*.py api/*.py

format:
	@echo "Formatting code with black..."
	uv run black logic/*.py cli/*.py api/*.py tests/*.py

test:
	@echo "Running tests with coverage..."
	uv run python -m pytest -v --cov=logic --cov=cli --cov=api --cov-report=term-missing

refactor: format lint
	@echo "Refactoring complete (format + lint)"

all: install format lint test
	@echo "All tasks completed successfully!"

run-api:
	@echo "Starting FastAPI application..."
	@echo "Visit http://localhost:8000 after startup"
	uv run python -m api.api

run-cli:
	@echo "Image Classification CLI"
	@echo ""
	@echo "Available commands:"
	@echo "  predict   - Predict image class"
	@echo "  resize    - Resize an image"
	@echo "  grayscale - Convert to grayscale"
	@echo "  normalize - Get image statistics"
	@echo "  crop      - Crop an image"
	@echo ""
	@echo "Example usage:"
	@echo "  uv run python -m cli.cli predict --image photo.jpg --seed 42"
	@echo "  uv run python -m cli.cli resize input.jpg 224 224 --output resized.jpg"

clean:
	@echo "Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".coverage" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	@echo "Cleanup complete"