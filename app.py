import streamlit as st
import pickle
import re
import nltk

nltk.download('stopwords')
nltk.download('punkt')

#loading models
clf = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

def cleanResume(text):
    text = re.sub('http\S+\s*', ' ', text)  # remove URLs
    text = re.sub('RT|cc', ' ', text)  # remove RT and cc
    text = re.sub('#\S+', '', text)  # remove hashtags
    text = re.sub('@\S+', '  ', text)  # remove mentions
    text = re.sub('[%s]' % re.escape("""!"$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', text)  # remove punctuations
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # remove non-ASCII characters
    text = re.sub('\s+', ' ', text)  # remove extra whitespace
    return text

# web app

def main():
    st.title("Resume Screening App")
    uploaded_file = st.file_uploader("Upload Resume", type=["pdf", "docx", "txt"])

    if uploaded_file is not None:
        try:
            resume_bytes = uploaded_file.read()
            resume_text = resume_bytes.decode('utf-8', errors='ignore')

        except UnicodeDecodeError:
            # if utf-8 decoding fails, try latin-1
            resume_text = resume_bytes.decode('latin-1', errors='ignore')

        cleaned_resume = cleanResume(resume_text)
        vectorized_resume = tfidf.transform([cleaned_resume])
        prediction = clf.predict(vectorized_resume)

        st.write("Prediction:", prediction[0])

        #map categories to job roles
        category_mapping = {
            0: "Advocate",
            1: "Arts",
            2: "Automation Testing",
            3: "Blockchain",
            4: "Business Analyst",
            5: "Civil Engineer",
            6: "Data Science",
            7: "Database Admin",
            8: "DevOps Engineer",
            9: "Dotnet Developer",
            10: "ETL Developer",
            11: " Electrical Engineer",
            12: "HR",
            13: " Hadoop",
            14: "Health and Fitness",
            15: "Java Developer",
            16: "Mechanical Engineer",
            17: "Network Security Engineer",
            18: "Operations Manager",
            19: "PMO",
            20: "Python Developer",
            21: "SAP Developer",
            22: "Sales Manager",
            23: "Testing",
            24: "Web Designer",
    
        }

        category_name = category_mapping.get(prediction[0], "Unknown Category")
        st.write("Predicted Job Role:", category_name)
#python main
if __name__ == '__main__':
    main()