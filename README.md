# PrivAI – A Privacy-Focused AI Assistant

**PrivAI** is a multimodal AI assistant designed to ensure privacy while offering intelligent responses from text, images, PDFs, and even YouTube videos. Built with future readiness for local LLM integration, it currently uses cloud-based models like Gemini for powerful performance.

---

## 🔐 Project Vision

PrivAI is built to:
- 🧠 Support multiple input types (Text, Image, PDF, YouTube URLs)
- 🌐 Use cloud-based LLMs temporarily (e.g., Google Gemini API)
- 💻 Be locally deployable in the future with on-device LLMs
- 🔒 Prioritize data privacy and user control

---

## 🛠️ Features

- 📄 Text-based query support  
- 🖼️ Image understanding (captioning, question-answering)  
- 📚 PDF content analysis and summarization  
- 📹 YouTube link summarization  
- 📥 Upload multiple file types  
- 🌐 Modular architecture for future local AI integration  

---

## 🧱 Tech Stack

| Technology     | Purpose                        |
|----------------|--------------------------------|
| Python         | Backend logic                  |
| Flask          | API server                     |
| Google Gemini  | Cloud-based LLM API            |
| HTML/CSS/JS    | Frontend interface             |
| OpenCV/PyMuPDF | Image & PDF handling           |
| SQLite         | Session logging (optional)     |

---

## 🧩 System Architecture

```
[User]
   |
   |----> [PrivAI Input Module]
                     |
        +------------+------------+
        |            |            |
   [Text]        [Image]       [PDF/Video]
        |            |            |
        +------------v------------+
                [Gemini API]
                     |
             [Output Generator]
                     |
                 [User Output]
```

---

## 🧪 How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/privai.git
   cd privai
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add your Gemini API Key**
   - Create a `.env` file or insert directly in `config.py`.

4. **Run the application**
   ```bash
   python app.py
   ```

---

## 🖼️ Sample Inputs Supported

- Text query like: *"Explain blockchain simply"*
- Image upload (e.g., flowcharts or diagrams)
- PDF upload (e.g., class notes)
- YouTube video link (e.g., tutorial summarization)

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙋‍♂️ Author

**Yugansh Gupta**  
Roll No: 08813702022  
Bachelor of Computer Applications  
Institute of Information Technology and Management

---

## 🌟 Future Scope

- On-device LLM inference using `llama.cpp`  
- Voice input/output support  
- Secure offline mode  
- Plugin-style extension for 3rd-party models

---
