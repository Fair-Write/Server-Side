# FairWrite (Server-Side)

FairWrite is a web-based text editor that provides grammar and gender-fair revisions. Built with FastAPI, it integrates with powerful NLP tools like `spaCy` and `language-tool-python` to enhance writing for clarity, accuracy, and inclusivity.

## Features

- **Grammar Check**: Automatically detect and suggest corrections for grammatical errors.
- **Gender-Fair Language**: Helps users adopt gender-neutral language and avoid biased terms.

## Tech Stack

- **FastAPI**: Web framework for building APIs quickly and efficiently.
- **Uvicorn**: ASGI server for serving FastAPI apps.
- **Pydantic**: Data validation and settings management.
- **Language Tool**: Grammar checking and text analysis.
- **spaCy**: NLP library for processing text, including tokenization and named entity recognition.

## Installation

Follow these steps to get the project up and running:

### Step 1: Clone the repository

Clone this project to your local machine:

```bash
git clone https://github.com/yourusername/fairwrite.git
cd fairwrite
```

### Step 2: Install dependencies

Install all required packages using `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Step 3: Download spaCy Model

Download the English model for spaCy:

```bash
python -m spacy download en_core_web_sm
```

### Step 4: Run the Application

Start the web app locally with Uvicorn:

```bash
uvicorn main:app --port 80
```

The app will be available at [http://127.0.0.1:80/](http://127.0.0.1:80/).

## Frontend Integration

- [FairWriteSide](https://github.com/Fair-Write/Client-Side.git): The official frontend for FairWrite, providing an intuitive interface for interacting with the backend API.
