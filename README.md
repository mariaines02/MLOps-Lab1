# MLOps-Lab1: Image Classification API

[![CI Pipeline](https://github.com/YOUR_USERNAME/MLOps-Lab1/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/MLOps-Lab1/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A machine learning project for image classification and preprocessing with Continuous Integration using GitHub Actions. This is the first stage of a three-lab series that will culminate in a deep learning image classifier.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Project Structure](#-project-structure)
- [Features](#-features)
- [Setup](#-setup)
- [Usage](#-usage)
- [Development](#-development)
- [CI/CD Pipeline](#-cicd-pipeline)
- [Testing](#-testing)

---

## 🎯 Project Overview

This project implements:
- **Image prediction** (currently random, will be ML-based in Lab3)
- **Image preprocessing** (resize, grayscale, crop, normalize)
- **CLI interface** for command-line interaction
- **REST API** using FastAPI for web integration
- **Automated CI/CD** with GitHub Actions

### Technology Stack

- **Language:** Python 3.11+
- **CLI Framework:** Click
- **API Framework:** FastAPI + Uvicorn
- **Image Processing:** Pillow (PIL)
- **Testing:** Pytest + Pytest-Cov
- **Code Quality:** Pylint + Black
- **Package Manager:** UV

---

## 📁 Project Structure

```
MLOps-Lab1/
├── logic/
│   ├── __init__.py
│   └── predictor.py          # ML logic and image processing
├── cli/
│   ├── __init__.py
│   └── cli.py                # Command-line interface
├── api/
│   ├── __init__.py
│   └── api.py                # FastAPI application
├── tests/
│   ├── __init__.py
│   ├── test_logic.py         # Unit tests for logic
│   ├── test_cli.py           # Integration tests for CLI
│   └── test_api.py           # Integration tests for API
├── .github/
│   └── workflows/
│       └── ci.yml            # GitHub Actions CI pipeline
├── Makefile                  # Task automation
├── pyproject.toml           # Project configuration
├── .gitignore
└── README.md
```

---

## ✨ Features

### 🎯 Prediction
- Predict image class from a set of predefined categories
- Random prediction with configurable seed (for now)
- Returns confidence score and all available classes

### 🖼️ Image Preprocessing

#### Resize
- Resize images to any dimensions
- Maintains image quality with LANCZOS resampling
- Supports both file and byte input

#### Grayscale Conversion
- Convert colored images to grayscale
- Useful for ML preprocessing

#### Crop
- Crop images to specific regions
- Specify exact coordinates (left, top, right, bottom)

#### Normalize
- Get image statistics (mean, min, max)
- Useful for data analysis and preprocessing

---

## 🚀 Setup

### Prerequisites

- Python 3.11 or higher
- UV package manager

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/YOUR_USERNAME/MLOps-Lab1.git
cd MLOps-Lab1
```

2. **Initialize environment:**
```bash
uv init
uv sync
```

3. **Install dependencies:**
```bash
make install
```

Or manually:
```bash
uv add pillow click fastapi uvicorn jinja2 httpx pytest pytest-cov pylint black python-multipart numpy
```

4. **Activate virtual environment:**
```bash
source .venv/bin/activate
```

---

## 💻 Usage

### Command Line Interface (CLI)

#### Predict Image Class
```bash
# Basic prediction
uv run python -m cli.cli predict

# With seed for reproducibility
uv run python -m cli.cli predict --seed 42

# With image path
uv run python -m cli.cli predict --image photo.jpg --seed 42
```

#### Resize Image
```bash
# Resize to 224x224
uv run python -m cli.cli resize input.jpg 224 224

# Resize and save
uv run python -m cli.cli resize input.jpg 224 224 --output resized.jpg
```

#### Convert to Grayscale
```bash
# Convert to grayscale
uv run python -m cli.cli grayscale input.jpg --output gray.jpg
```

#### Get Image Statistics
```bash
# Get normalization stats
uv run python -m cli.cli normalize input.jpg
```

#### Crop Image
```bash
# Crop image (left, top, right, bottom)
uv run python -m cli.cli crop input.jpg 10 10 200 200 --output cropped.jpg
```

#### CLI Help
```bash
# General help
uv run python -m cli.cli --help

# Command-specific help
uv run python -m cli.cli predict --help
uv run python -m cli.cli resize --help
```

### REST API

#### Start the API Server
```bash
# Using Makefile
make run-api

# Or directly
uv run python -m api.api
```

The API will be available at `http://localhost:8000`

#### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Home page with API information |
| `/health` | GET | Health check endpoint |
| `/docs` | GET | Interactive API documentation (Swagger UI) |
| `/predict` | POST | Predict image class |
| `/resize` | POST | Resize an image |
| `/grayscale` | POST | Convert image to grayscale |
| `/normalize` | POST | Normalize an image |
| `/crop` | POST | Crop an image |

#### Using the API

1. **Visit the home page:**
   - Open `http://localhost:8000` in your browser

2. **Interactive documentation:**
   - Go to `http://localhost:8000/docs`
   - Click on any endpoint
   - Click "Try it out"
   - Upload an image file
   - Click "Execute"

3. **Example with curl:**

```bash
# Predict image class
curl -X POST "http://localhost:8000/predict?seed=42" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.jpg"

# Resize image
curl -X POST "http://localhost:8000/resize?width=224&height=224" \
  -H "accept: application/json" \
  -F "file=@image.jpg" \
  --output resized.jpg

# Normalize image
curl -X POST "http://localhost:8000/normalize" \
  -H "accept: application/json" \
  -F "file=@image.jpg" \
  --output normalized.jpg
```

---

## 🛠️ Development

### Makefile Commands

```bash
# Show all available commands
make help

# Install all dependencies
make install

# Format code with Black
make format

# Lint code with Pylint
make lint

# Run tests with coverage
make test

# Format + Lint
make refactor

# Run all checks (install, format, lint, test)
make all

# Clean cache and temporary files
make clean

# Run API server
make run-api

# Show CLI help
make run-cli
```

### Code Quality Standards

- **Formatting:** Black (line length: 100)
- **Linting:** Pylint (minimum score: 9.0/10)
- **Testing:** Pytest with coverage
- **Type Hints:** Encouraged for better code clarity

---

## 🔄 CI/CD Pipeline

The project uses **GitHub Actions** for Continuous Integration.

### Pipeline Steps

1. **Checkout code** - Clone repository
2. **Setup Python** - Install Python 3.11
3. **Install UV** - Install UV package manager
4. **Install dependencies** - Install all required packages
5. **Format check** - Verify code formatting with Black
6. **Lint** - Run Pylint (must score ≥9.0)
7. **Test** - Run all tests with coverage

### Pipeline Triggers

- **Push** to `main` or `develop` branches
- **Pull requests** to `main` or `develop` branches

### Status Badge

The CI status badge at the top of this README shows the current state of the pipeline:
- ✅ **Green:** All checks passed
- ❌ **Red:** One or more checks failed

### Viewing Pipeline Results

1. Go to the **Actions** tab in your GitHub repository
2. Click on a workflow run to see details
3. Expand each step to view logs

---

## 🧪 Testing

### Test Structure

- **Unit Tests** (`test_logic.py`): Test individual functions in the logic module
- **CLI Integration Tests** (`test_cli.py`): Test CLI commands end-to-end
- **API Integration Tests** (`test_api.py`): Test API endpoints

### Running Tests

```bash
# Run all tests
make test

# Run tests with verbose output
uv run python -m pytest -v

# Run specific test file
uv run python -m pytest tests/test_logic.py -v

# Run with coverage report
uv run python -m pytest --cov=logic --cov=cli --cov=api --cov-report=html
```

### Test Coverage

The project aims for high test coverage (>90%) across all modules.

Coverage report locations:
- **Terminal:** Displayed after `make test`
- **HTML:** Generated in `htmlcov/` directory

### Test Fixtures

The tests use pytest fixtures for:
- Creating test images
- Setting up test clients (CLI runner, API client)
- Providing reusable test data


## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👤 Author

**[Maria Ines Haddad]**
- GitHub: [@mariaines02](https://github.com/mariaines02)
- Course: MLOps - Master's in Machine Learning

---
