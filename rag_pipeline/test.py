from tqdm import tqdm
import os
from documents import DocumentStore
from utils import Course, Certificate
from rag import BasicRAG

# List of course numbers (from Adobe certification site)
course_numbers = [
    1043, 1045, 1046, 1047, 1048, 1049, 1050, 1054, 1055, 1056, 
    1057, 1058, 1059, 1060, 1061, 1062, 1063, 1064, 1065, 1066, 
    1067, 1068, 1069, 1221
]

# Initialize document store
store = DocumentStore(similarity_metric="cosine", storage_path="./chroma_storage")

"""
# Process Courses
print("Processing Courses...")
courses = []
for course_id in tqdm(course_numbers):
    try:
        course = Course(f'https://certification.adobe.com/courses/{course_id}')
        store.add_document(course)  # Store course in ChromaDB
        courses.append(course)
    except Exception as e:
        print(f"Error processing course {course_id}: {e}")

print(f"Successfully added {len(courses)} courses.")

# Process Certificates
certificate_htmls_location = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dependency_graph/certificate_htmls'))
certificates = []

print("Processing Certificates...")
for html in tqdm(os.listdir(certificate_htmls_location)):
    try:
        certificate = Certificate(f'{certificate_htmls_location}/{html}')
        store.add_document(certificate)  # Store certificate in ChromaDB
        certificates.append(certificate)
    except Exception as e:
        print(f"Error processing certificate {html}: {e}")

print(f"Successfully added {len(certificates)} certificates.")
"""

# Run RAG pipeline to test retrieval & OpenAI response
rag = BasicRAG(document_store=store)
query = "I'd like to learn more about the Adobe Commerce suite of courses and certificates. What is available for me?"
response = rag.run_rag_pipeline(query)

print("\nGenerated Response:")
print(response)