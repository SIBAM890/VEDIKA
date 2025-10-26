Vedika - AI Assistant for Sri Sri University

Vedika is an intelligent, conversational AI assistant designed to answer questions and provide information about Sri Sri University (SSU). It leverages Retrieval-Augmented Generation (RAG) powered by OpenAI and LangChain, using data scraped directly from the official university website.

✨ Features

Conversational AI: Engage in natural, human-like conversations for casual chat.

University Knowledge Base: Answers specific questions about SSU admissions, courses, fees, facilities, faculty, and more using data scraped from the official website.

Intelligent Routing: Automatically detects user intent to switch between casual chat, providing specific university information (RAG), listing all courses, or handling simple greetings/thanks.

Retrieval-Augmented Generation (RAG): Uses OpenAI embeddings (text-embedding-3-small) and FAISS vector search to find relevant information from the scraped website content before generating answers.

Advanced Web Crawler: Intelligently scrapes the entire srisriuniversity.edu.in website, including handling HTML, extracting text from PDFs, and saving other assets, to build a comprehensive knowledge base.

Voice Interaction: Supports voice input via microphone (Speech-to-Text) and speaks responses aloud (Text-to-Speech using a designated voice), especially when input is via voice.

Suggestion Chips: Provides interactive starting points for users.

Branded UI: Features a professional user interface themed with Sri Sri University's official logo and colors.

🛠️ Tech Stack

Backend:

Language: Python 3.10+

Web Framework: FastAPI

Server: Uvicorn

AI Orchestration: LangChain

LLM & Embeddings: OpenAI (gpt-4o-mini for generation, text-embedding-3-small for embeddings)

Vector Store: FAISS (CPU version)

Web Scraping: Requests, BeautifulSoup4, PyPDF2

Configuration: python-dotenv

Frontend:

Structure: HTML5

Styling: CSS3

Logic: JavaScript (ES6+)

Speech: Web Speech API (SpeechRecognition, SpeechSynthesis)

Icons: Font Awesome

🚀 Getting Started

Follow these steps to set up and run the Vedika chatbot locally.

Prerequisites:

Python 3.10 or higher installed.

Git installed.

An OpenAI API key.

1. Clone the Repository:

git clone [https://github.com/SIBAM890/VEDIKA.git](https://github.com/SIBAM890/VEDIKA.git)
cd VEDIKA


2. Set Up Python Environment:
Create and activate a virtual environment:

python -m venv .venv
# On Windows (PowerShell/Git Bash)
.\.venv\Scripts\Activate.ps1
# On macOS/Linux
# source .venv/bin/activate


Install the required dependencies:

pip install -r requirements.txt


3. Configure API Keys:
Create a file named .env in the root VEDIKA directory. Add your OpenAI API key:

OPENAI_API_KEY="sk-..."


(Optionally, you can also set SSU_ROOT_URL if you want to scrape a different starting point, though the default is correct).

4. Build the Knowledge Base (Ingestion):
Run the ingestion script. This will crawl the website (based on backend/config.py settings) and build the FAISS vector store using OpenAI embeddings. This step can take a significant amount of time.

python ingest.py


Monitor the terminal logs. Upon completion, you should see files (index.faiss, index.pkl) inside the vector_store/faiss_index/ directory. Downloaded PDFs, images, etc., will be in data/raw/.

5. Run the Backend Server:
Start the FastAPI server using Uvicorn. Keep this terminal running.

uvicorn backend.main:app --reload
