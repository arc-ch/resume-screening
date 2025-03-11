import streamlit as st
import os
import pickle
import docx
import PyPDF2
import re
from google import genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure Gemini API
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("GEMINI_API_KEY not found! Make sure .env is correctly set up.")
client = genai.Client(api_key=api_key)


# Load models and vectorizer
svc_model = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))
le = pickle.load(open('encoder.pkl', 'rb'))

# Function to clean resume text
def cleanResume(txt):
    cleanText = re.sub(r'http\S+\s', ' ', txt)
    cleanText = re.sub(r'RT|cc', ' ', cleanText)
    cleanText = re.sub(r'#\S+\s', ' ', cleanText)
    cleanText = re.sub(r'@\S+', ' ', cleanText)
    cleanText = re.sub(r'[%s]' % re.escape("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub(r'\s+', ' ', cleanText)
    return cleanText

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

# Function to extract text from DOCX
def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ''
    for paragraph in doc.paragraphs:
        text += paragraph.text + '\n'
    return text

# Function to extract text from TXT with explicit encoding handling
def extract_text_from_txt(file):
    try:
        text = file.read().decode('utf-8')
    except UnicodeDecodeError:
        text = file.read().decode('latin-1')  # Fallback encoding
    return text

# Function to handle file upload and extraction
def handle_file_upload(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension == 'pdf':
        return extract_text_from_pdf(uploaded_file)
    elif file_extension == 'docx':
        return extract_text_from_docx(uploaded_file)
    elif file_extension == 'txt':
        return extract_text_from_txt(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF, DOCX, or TXT file.")

# Predict job category
def predict_category(resume_text):
    cleaned_text = cleanResume(resume_text)
    vectorized_text = tfidf.transform([cleaned_text]).toarray()
    predicted_category = svc_model.predict(vectorized_text)
    return le.inverse_transform(predicted_category)[0]

# Construct prompt for Gemini resume scoring
def construct_resume_score_prompt(resume, job_description):
    return f'''Act as an HR Manager with 20 years of experience. 
Compare the provided resume with the job description and rate the resume based on skill match.
Provide the following output **using Markdown formatting** with no unnecessary introductory sentences or confirmations:
- **Score**: The score out of 100.
- **Strengths**: A brief list of strengths based on the job description.
- **Weaknesses**: A brief list of weaknesses or areas for improvement based on the job description.
- **Recommendations**: Suggestions for improvement (e.g., skills to develop, experience to add, etc.).

Here is the Resume: {resume}
Here is the Job Description: {job_description}

Response format (use **Markdown** consistently for output):
- **Score**: <score>
- **Strengths**: <list of strengths>
- **Weaknesses**: <list of weaknesses>
- **Recommendations**: <list of recommendations>'''


def get_resume_score(resume, job_description):
    prompt = construct_resume_score_prompt(resume, job_description)
    
    response = client.models.generate_content(
        model="gemini-2.0-flash", 
        contents=[prompt]  # Pass the prompt as the content
    )
    
    # Extract and return the score from the response text
    return response.text.split('score:')[-1].strip()

def gemini_chat(prompt):
    
    resume_keywords = [
        'resume', 'cv', 'cover letter', 'job application', 'skills', 'experience', 
        'education', 'work history', 'career', 'professional profile', 'improvement tips', 
        'job interview', 'job search', 'career development', 'job skills', 'work experience', 
        'resume tips', 'career goals', 'job opportunities', 'professional growth', 
        'employment', 'job description', 'job role', 'career advice', 'resume formatting', 
        'networking', 'interview preparation', 'resume writing', 'career path', 'salary expectations',
        'salary negotiation', 'professional experience', 'job offer', 'career transition'
    ]

    prompt_lower = prompt.lower()

    # Check if  prompt contains any of the resume or improvement-related keywords
    if not any(keyword in prompt_lower for keyword in resume_keywords):
        return "Please refrain from asking random questions and stick to questions related to resumes, job applications, improvement tips, career development, and related topics."
    
    # Generate content using the corrected API method
    response = client.models.generate_content(
        model="gemini-2.0-flash",  # You can change the model to whatever you prefer
        contents=[prompt]  # Pass the user input as the content
    )
    
    # Return the response text generated by the API
    return response.text


# Streamlit App
def main():
    st.set_page_config(page_title="Resume Screening", page_icon="📄", layout="wide")
    
    # Add some custom CSS for a better look and feel
    st.markdown("""
    <style>
        .big-font {
            font-size: 30px;
            font-weight: bold;
            color: #FFFFFF;  /* WHITE color for the category */
        }
        .title-font {
            font-size: 35px;
            font-weight: 700;
            color: #FF6347;  /* Tomato color for the title */
        }
        .button-style {
            background-color: #FF5722;  /* A bright orange color for buttons */
            color: white;
            font-size: 16px;
        }
        .subheader {
            color: #2196F3;  /* Blue color for subheaders */
        }
        .chatbox {
            border: 2px solid #FFEB3B;  /* Yellow border for the chatbox */
            border-radius: 8px;
        }
                
        .chat-response {
            font-size: 18px;  /* Adjust the font size for Gemini's response */
            color: #FF6347;  /* Tomato Red color for Gemini's response */
            font-weight: normal;
        }
        .chat-header {
            font-size: 24px;  /* Font size for the 'Chat with Gemini' header */
            font-weight: bold;
            color:#FFDB58;  /* Tomato Red color for the header */
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Title with an emoji
    st.markdown('<h1 class="title-font">🚀 Resume Screening & Chat Assistant 📄</h1>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload a Resume 📑", type=["pdf", "docx", "txt"])

    if uploaded_file:
        # file_type = uploaded_file.name.split(".")[-1].lower()
       # resume_text = extract_text(uploaded_file, file_type)
        resume_text = handle_file_upload(uploaded_file)
        st.success("✅ Resume text extracted successfully!")
        
        # Display extracted resume text (optional)
        if st.checkbox("Show extracted text 📝"):
            st.text_area("Resume Content", resume_text, height=300, key="resume")

        # Predict job category
        job_category = predict_category(resume_text)

        # Display the predicted job category in larger, bold font with some color
        st.markdown(f'<p class="big-font">🔍 Predicted Job Category: <span style="color: #4CAF50;">{job_category} 🧑‍💼</span></p>', unsafe_allow_html=True)

# Get resume score with a button styled in orange
        if st.button("Get Resume Score 💯", key="score", help="Click to get the resume match score.", use_container_width=True):
            score = get_resume_score(resume_text, job_category)
            st.session_state.resume_score = score  # Save score in session state

    # Display Resume Score only if it exists in session state
    if 'resume_score' in st.session_state:
        # st.subheader("Resume Score 🏆")
        st.markdown('<p class="chat-header">Resume Score 🏆</p>', unsafe_allow_html=True)

        st.write(st.session_state.resume_score)  # Display saved score

    # st.subheader("Chat with Gemini 🤖", anchor="chat")
    st.markdown('<p class="chat-header">💬 Chat with Gemini 🤖</p>', unsafe_allow_html=True)
    user_input = st.text_area("Ask anything about resumes, improvement tips, etc. 💬 ( Type **IN SHORT** for Concise Output) ", key="chat_input", help="Type your question here.")


    if st.button("Send 📨", key="send", help="Click to send your message to Gemini.", use_container_width=True):
        response = gemini_chat(user_input)
        st.session_state.chat_response = response  # Save response in session state
        #st.write(response)

    # Display Chat Response only if it exists in session state
    if 'chat_response' in st.session_state:
        # st.subheader("Gemini Response 🗣️")
        st.markdown('<p class="chat-header"> Gemini Response 🗣️</p>', unsafe_allow_html=True)
        st.write(st.session_state.chat_response)  # Display saved chat response

               # Example Questions related to resumes (added Markdown)
    st.markdown("""
    ### Example Queries 📋:
    ```python
    1. Which is the most common font name and size in resumes?
    2. How do I include leadership experience if I haven't had a formal leadership role?
    3. What are the best skills to highlight for entry-level jobs?
    4. Can I put volunteer work on my resume?
    5. How do I quantify my accomplishments in a student project?
   ```
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()