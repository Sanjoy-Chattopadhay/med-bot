# 🧠 MedBot GEN-AI

A domain-specific **Medical Science Chatbot** powered by **Generative AI** using **RAG (Retrieval-Augmented Generation)**.  
Built with **LangChain**, **Groq's LLaMA 3 API**, and **FAISS** to provide accurate, context-grounded responses directly from a medical textbook.

---

## 🚀 Features

- 📄 Ingests medical textbooks (PDF)
- 🔍 Chunking & semantic search with **FAISS**
- 🧠 Uses **Groq-hosted LLaMA 3** for fast inference
- ⚙️ Built using **LangChain RAG pipeline**
- 💬 Ready for CLI or Streamlit-based chat interface

---

## 🛠️ Tech Stack

- **Language**: Python 3.10+
- **LLM Inference**: [Groq API](https://console.groq.com/)
- **RAG Framework**: LangChain
- **Embeddings & Search**: FAISS
- **Frontend (Optional)**: Streamlit

---

## 📦 Setup & Installation

```bash
# 1. Clone the repository
git clone https://github.com/rohit0796/MedBot-GEN-AI.git
cd MedBot-GEN-AI

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your Groq API key
echo "GROQ_API_KEY=your_groq_api_key" > .env

# 4. Run the application (CLI)
python app.py

# Optional: Run with Streamlit
streamlit run app.py
```
## 🤝 Contributions
Feel free to fork, open issues, or submit pull requests.
If you're interested in Generative AI or Healthcare AI, let's connect!

## 🙋‍♂️ About Me
- Rohit Kumar Sahu
- 👨‍🎓 M.Tech CSE @ NIT Durgapur
- 📬 rrohitkumarsahu2002@gmail.com
