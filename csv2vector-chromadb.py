import pandas as pd

df = pd.read_csv("hf://datasets/QuyenAnhDE/Diseases_Symptoms/Diseases_Symptoms.csv")
df['Symptoms'] = df['Symptoms'].str.split(',')
df['Symptoms'] = df['Symptoms'].apply(lambda x: [s.strip() for s in x])
from sentence_transformers import SentenceTransformer
import chromadb
# Load the embedding model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
client = chromadb.PersistentClient(path='./chromadb')
collection = client.get_or_create_collection(name="symptoms")
query = 'dizziness'

# Generate embedding for the query
query_embedding = model.encode([query])

# Perform similarity search in ChromaDB
results = collection.query(
    query_embeddings=query_embedding.tolist(),
    n_results=5  # Return top 5 similar symptoms
)

# Display the closest matching symptoms
print("Matching symptoms:")
for doc in results['documents']:
    print(doc)

matching_symptoms=results['documents'][0]
matching_diseases = df[df['Symptoms'].apply(lambda x: any(s in matching_symptoms for s in x))]
# Display matching diseases and their full symptoms
for index, row in matching_diseases.iterrows():
    print(f"Disease: {row['Name']}")
    print(f"Symptoms: {', '.join(row['Symptoms'])}")
    print(f"Treatments: {row['Treatments']}")
    print()

disease_list = []
for index, row in matching_diseases.iterrows():
    disease_info = {
        'Disease': row['Name'],
        'Symptoms': row['Symptoms'],
        'Treatments': row['Treatments']
    }
    disease_list.append(disease_info)

print(disease_list)