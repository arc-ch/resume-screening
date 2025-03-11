# 🚀 Resume Screening & Chat Assistant 📄\

![file_2025-03-11_17 10 23](https://github.com/user-attachments/assets/66d205d7-3808-4564-b8d3-bdcc46320e76)


## 📌 Overview
This is a **Resume Screening & Chat Assistant** built using **Streamlit**, **Machine Learning**, and **Google Gemini AI**. It helps users analyze resumes, predict job categories, and evaluate how well a resume matches a job description. Additionally, it provides an AI-powered chat assistant for resume-related queries.

## 🚀 Features

### 🎯 Resume Screening
- Upload resumes in **PDF, DOCX, or TXT** formats.
- Extract text automatically and clean it for analysis.
- Predict job category using **TF-IDF vectorization** and **Logistic Regression**.
- Provide an AI-generated **Resume Score** based on job description matching.

### 💬 AI Chat Assistant
- Chat with **Google Gemini AI** for resume improvement suggestions.
- Get tips on career development, job applications, and professional growth.
- Filters out non-resume-related queries:-
  
  ![file_2025-03-11_17 12 22](https://github.com/user-attachments/assets/6c147b34-8795-4749-ad40-961e48be6213)


## 🛠️ Tech Stack
- **Python**
- **Streamlit** for UI
- **Scikit-learn** for ML
- **Google Gemini AI** for NLP
- **TF-IDF Vectorization** for text processing
- **Pickle** for model storage

## 📂 File Structure
```bash
📂 Resume-Screening-Assistant
├── app.py                # Main Streamlit application
├── clf.pkl               # Trained Support Vector Classifier model (optional)
├── logreg_model.pkl      # Trained Logistic Regression model
├── tfidf.pkl             # TF-IDF Vectorizer
├── encoder.pkl           # Label Encoder
├── requirements.txt      # Required dependencies
├── .env                  # Environment variables (GEMINI API key)
```

## 📥 Installation & Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/resume-screening.git
   cd resume-screening
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your `.env` file:
   ```bash
   echo "GEMINI_API_KEY=your_api_key" > .env
   ```

4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## 🎯 How It Works
1. **Upload a resume** 📑
2. **Extract text automatically** 📝
3. **Predict job category** 🔍
4. **Get resume score based on job description** 💯
5. **Chat with AI for improvements** 🤖

## 📌 Example Queries
```plaintext
1. Which is the most common font name and size in resumes?
2. How do I include leadership experience if I haven't had a formal leadership role?
3. What are the best skills to highlight for entry-level jobs?
4. Can I put volunteer work on my resume?
5. How do I quantify my accomplishments in a student project?
```

## 🏆 Future Enhancements
- Add **deep learning models** for better job category predictions.
- Improve **resume scoring accuracy** with more AI-driven insights.
- Implement **ATS compliance checks** for resumes.

## 🤝 Contributing
Pull requests are welcome! Feel free to open an issue for bug fixes or new feature suggestions.

## 📜 License
This project is **MIT Licensed**.

---
🚀 Built with ❤️ by [Your Name](https://github.com/yourusername)
