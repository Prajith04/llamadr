
# Load the dataset
import pandas as pd

df = pd.read_csv("hf://datasets/QuyenAnhDE/Diseases_Symptoms/Diseases_Symptoms.csv")

# Split the Symptoms column into individual symptoms
# Assuming symptoms are comma-separated
df['Symptoms'] = df['Symptoms'].str.split(',')
df['Symptoms'] = df['Symptoms'].apply(lambda x: [s.strip().lower() for s in x])

# Flatten the list of symptoms and strip whitespace
symptoms = df['Symptoms'].explode().str.strip()

# Get unique symptoms
unique_symptoms = symptoms.unique()

print(f"Unique symptoms: {unique_symptoms}")

from sentence_transformers import SentenceTransformer
import chromadb
# Load the embedding model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Initialize ChromaDB client and create a collection for symptoms
#client = chromadb.Client()
client = chromadb.PersistentClient(path='./chromadb')
collection = client.create_collection('symptomsvectordb')

# Generate embeddings for the unique symptoms
symptom_embeddings = model.encode(unique_symptoms)

#Store embeddings in ChromaDB
for i, symptom in enumerate(unique_symptoms):
    collection.add(
        documents=[symptom],
        embeddings=[symptom_embeddings[i].tolist()],
        ids=[f"symptom_{i}"]
    )

# Example user query
query = 'vomiting'

# Generate embedding for the query
query_embedding = model.encode([query])

# Perform similarity search in ChromaDB
results = collection.query(
    query_embeddings=query_embedding.tolist(),
    n_results=1  # Return top 5 similar symptoms
)

# Display the closest matching symptoms
print("Matching symptoms:")
for doc in results['documents']:
    print(doc)

matching_symptoms=results['documents'][0]

matching_diseases = df[df['Symptoms'].apply(lambda x: any(s.lower() in matching_symptoms for s in x))]

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