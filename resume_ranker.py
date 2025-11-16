import spacy
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import csv

csv_filename = "ranked_resumes.csv"

# Load spaCy NER model
nlp = spacy.load("en_core_web_sm")

# Sample job description
job_description = """jvm developer."""

# List of resume PDF file paths
resume_paths = ["resume1.pdf", "resume2.pdf", "resume3.pdf"]  # Add more file paths here

# Extract text from PDFs
def extract_text_from_pdf(pdf_path):
    try:
        with open(pdf_path, "rb") as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
            return text
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""

# Extract emails, names, and locations using spaCy NER
def extract_entities(text):
    emails = set(re.findall(r'\S+@\S+', text))
    doc = nlp(text)

    names = []
    locations = []

    for ent in doc.ents:
        if ent.label_ == "PERSON":
            names.append(ent.text)
        elif ent.label_ == "GPE":  # GPE (Geopolitical Entity) includes locations
            locations.append(ent.text)

    return list(emails), names, locations

# Extract job description features using TF-IDF
tfidf_vectorizer = TfidfVectorizer()
job_desc_vector = tfidf_vectorizer.fit_transform([job_description])

# Rank resumes based on similarity
ranked_resumes = []
for resume_path in resume_paths:
    resume_text = extract_text_from_pdf(resume_path)
    emails, names, locations = extract_entities(resume_text)
    resume_vector = tfidf_vectorizer.transform([resume_text])
    similarity = cosine_similarity(job_desc_vector, resume_vector)[0][0]
    ranked_resumes.append((names, emails, similarity, locations))

# Sort resumes by similarity score
ranked_resumes.sort(key=lambda x: x[2], reverse=True)

# Display ranked resumes with emails, names, and locations
for rank, (names, emails, similarity, locations) in enumerate(ranked_resumes, start=1):
    name = names[0] if names else "N/A"
    email = emails[0] if emails else "N/A"
    location = locations[0] if locations else "N/A"
    print(f"Rank {rank}: Name: {name}, Email: {email}, Location: {location}, Similarity: {similarity:.2f}")

# Write results to CSV
with open(csv_filename, "w", newline="") as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["Rank", "Name", "Email", "Location", "Similarity"])
    
    for rank, (names, emails, similarity, locations) in enumerate(ranked_resumes, start=1):
        name = names[0] if names else "N/A"
        email = emails[0] if emails else "N/A"
        location = locations[0] if locations else "N/A"
        csv_writer.writerow([rank, name, email, location, similarity])
