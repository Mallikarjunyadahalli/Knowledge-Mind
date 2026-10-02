# 🧠 Knowledge Mind

**Chat with your documents.** Knowledge Mind is a Retrieval-Augmented Generation (RAG) app. Upload PDFs or text files, ask questions in a ChatGPT-style interface, and get answers grounded in your own documents, with sources shown for every answer.

![Knowledge Mind screenshot](assets/screenshot.png)

> Add a screenshot of your app at `assets/screenshot.png`, or delete the line above.

---

## ✨ Features

- **Chat interface** with streaming answers and message history
- **Multiple documents:** upload several PDF, TXT, or MD files at once
- **Source citations:** every answer shows the document name, page number, and the matching text
- **Follow-up questions:** the app remembers the recent conversation
- **Model choice:** switch between GPT-OSS and Llama models on Groq
- **Adjustable settings:** number of sections searched, creativity, and a sources toggle
- **Quick-start prompts:** one-click questions like "Summarize" and "Key points"
- **Export chat:** download the conversation as Markdown
- **Grounded answers:** if the answer isn't in your documents, the app says so instead of guessing

---

## 🛠️ Tech Stack

| Part            | Tool                                      |
| --------------- | ----------------------------------------- |
| Interface       | [Streamlit](https://streamlit.io)         |
| RAG framework   | [LangChain](https://www.langchain.com)    |
| LLM             | [Groq](https://groq.com) (GPT-OSS, Llama) |
| Vector database | [ChromaDB](https://www.trychroma.com)     |
| Embeddings      | Configured in `src/embeddings.py`         |

---

## ⚙️ How It Works

```
Upload files → Load text → Split into chunks → Create embeddings → Store in Chroma
                                                                        │
Your question → Find the most similar chunks ───────────────────────────┘
                          │
                          ▼
         Chunks + question + chat history → LLM (Groq) → Answer with sources
```

1. **Load:** PDFs and text files are read and converted to text.
2. **Split:** the text is cut into small overlapping chunks.
3. **Embed:** each chunk becomes a vector and is stored in ChromaDB.
4. **Retrieve:** your question is matched against the stored chunks.
5. **Generate:** the best chunks are sent to the LLM, which writes an answer using only that information.

---

## 📁 Project Structure

```
knowledge-mind/
├── app.py                  # Streamlit app (UI + chat logic)
├── src/
│   ├── loader.py           # PDF and text loaders
│   ├── splitter.py         # Chunking
│   └── embeddings.py       # Embedding model
├── .streamlit/
│   └── config.toml         # Theme settings
├── data/documents/         # Uploaded files are saved here
├── .env                    # Your API key (not uploaded to GitHub)
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Mallikarjunyadahalli/Knowledge-Mind.git
cd knowledge-mind
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your Groq API key

Get a free key from the [Groq Console](https://console.groq.com), then create a `.env` file in the project root:

```env
GROQ_API_KEY=your_api_key_here
```

### 5. Run the app

```bash
streamlit run app.py
```

Open the link shown in your terminal (usually http://localhost:8501).

---

## 💡 How to Use

1. Upload one or more documents in the sidebar.
2. Select **Process documents** and wait for the "Documents ready" message.
3. Type a question in the chat box, or pick a quick-start prompt.
4. Open **Sources** under an answer to see where it came from.
5. Use **New chat** to start over, or **Remove all documents** to reset everything.

---

## 🔧 Settings

| Setting            | What it does                                                                 |
| ------------------ | ---------------------------------------------------------------------------- |
| Model              | Pick the Groq model used to write answers                                    |
| Sections to search | How many document chunks are sent to the model (more = more context, slower) |
| Creativity         | Lower is more factual, higher is more varied                                 |
| Show sources       | Show or hide the source cards under answers                                  |

---

## 🗺️ Roadmap

- [ ] Support for DOCX and web pages
- [ ] Save the vector database between sessions
- [ ] Ask questions about a single selected document
- [ ] Dark mode
- [ ] Deploy on Streamlit Community Cloud

---

## ⚠️ Notes

- Answers depend on the quality of the uploaded documents. Scanned PDFs without selectable text won't work unless you add OCR.
- Never upload your `.env` file to GitHub. Add it to `.gitignore`.

---

## 🤝 Contributing

Contributions are welcome. Fork the repo, create a branch, and open a pull request.

---

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## 👤 Author

**Mallikarjun**
GitHub: https://github.com/Mallikarjunyadahalli
