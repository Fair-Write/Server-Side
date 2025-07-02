# FairWrite (Server-Side)

FairWrite is a web-based text editor that provides grammar and gender-fair revisions. Built with FastAPI, it integrates with powerful NLP tools like `spaCy` and `language-tool-python` to enhance writing for clarity, accuracy, and inclusivity.

## ✨ Features

- **Grammar Check:** Detects and suggests corrections for grammatical errors.
- **Gender-Fair Language:** Encourages gender-neutral language and avoids biased terms.
- **Custom Preferred Pronoun:** Users can specify their preferred pronouns for personalized suggestions.
- **Admin Gender Terms Control:** Admins can manage and customize gender term revisions (CRUD support).

## 🛠️ Tech Stack

- **FastAPI:** Web framework for building APIs.
- **Uvicorn:** ASGI server for FastAPI apps.
- **Pydantic:** Data validation and settings management.
- **Language Tool:** Grammar checking and text analysis.
- **spaCy:** NLP library for text processing.

## 📦 Installation

Follow these steps to set up the project:

### 1. Install Python

Make sure you have **Python 3+** installed. You can download it from [python.org](https://www.python.org/downloads/).

### 2. Clone the Repository

```bash
git clone https://github.com/yourusername/fairwrite.git
cd fairwrite
```

### 3. Create and Activate a Virtual Environment

```bash
python -m venv venv
```

- **On macOS/Linux:**
     ```bash
     source venv/bin/activate
     ```
- **On Windows:**
     ```bash
     venv\Scripts\activate
     ```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Download spaCy Model

```bash
python -m spacy download en_core_web_sm
```

### 6. Run the Application

```bash
uvicorn main:app --port 80
```

The app will be available at [http://127.0.0.1:80/](http://127.0.0.1:80/).

---

## 🚀 Frontend Integration

See [FairWrite Client-Side](https://github.com/yourusername/fairwrite-client) for the official frontend interface.
