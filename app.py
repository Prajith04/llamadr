from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import requests
# Define FastAPI app
app = FastAPI()

origins = [
    "http://localhost:5173",
    "localhost:5173"
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Load the dataset and model at startup
df = pd.read_csv("hf://datasets/QuyenAnhDE/Diseases_Symptoms/Diseases_Symptoms.csv")
df['Symptoms'] = df['Symptoms'].str.split(',')
df['Symptoms'] = df['Symptoms'].apply(lambda x: [s.strip() for s in x])

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
client = chromadb.PersistentClient(path='./chromadb')
collection = client.get_or_create_collection(name="symptomsvector")

class SymptomQuery(BaseModel):
    symptom: str

# Endpoint to handle symptom query and return matching symptoms
@app.post("/find_matching_symptoms")
def find_matching_symptoms(query: SymptomQuery):
    # Generate embedding for the symptom query
    symptoms = query.symptom.split(',')
    all_results = []

    for symptom in symptoms:
        symptom = symptom.strip()
        query_embedding = model.encode([symptom])

        # Perform similarity search in ChromaDB
        results = collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=3  # Return top 3 similar symptoms for each symptom
        )
        all_results.extend(results['documents'][0])

    # Remove duplicates while preserving order
    matching_symptoms = list(dict.fromkeys(all_results))

    return {"matching_symptoms": matching_symptoms}

# Endpoint to handle symptom query and return matching diseases
@app.post("/find_matching_diseases")
def find_matching_diseases(query: SymptomQuery):
    # Generate embedding for the symptom query
    query_embedding = model.encode([query.symptom])

    # Perform similarity search in ChromaDB
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=5  # Return top 5 similar symptoms
    )

    # Extract matching symptoms
    matching_symptoms = results['documents'][0]

    # Filter diseases that match the symptoms
    matching_diseases = df[df['Symptoms'].apply(lambda x: any(s in matching_symptoms for s in x))]

    return {"matching_diseases": matching_diseases['Name'].tolist()}

# Endpoint to handle symptom query and return detailed disease list
@app.post("/find_disease_list")
def find_disease_list(query: SymptomQuery):
    # Generate embedding for the symptom query
    query_embedding = model.encode([query.symptom])

    # Perform similarity search in ChromaDB
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=5  # Return top 5 similar symptoms
    )

    # Extract matching symptoms
    matching_symptoms = results['documents'][0]

    # Filter diseases that match the symptoms
    matching_diseases = df[df['Symptoms'].apply(lambda x: any(s in matching_symptoms for s in x))]

    # Create a list of disease information
    disease_list = []
    symptoms_list = []
    unique_symptoms_list = []
    for _, row in matching_diseases.iterrows():
        disease_info = {
            'Disease': row['Name'],
            'Symptoms': row['Symptoms'],
            'Treatments': row['Treatments']
        }
        disease_list.append(disease_info)
        symptoms_info = row['Symptoms']
        symptoms_list.append(symptoms_info)
    for i in range(len(symptoms_list)):
        for j in range(len(symptoms_list[i])):
            if symptoms_list[i][j] not in unique_symptoms_list:
                unique_symptoms_list.append(symptoms_list[i][j])
    return {"disease_list": disease_list, "unique_symptoms_list": unique_symptoms_list}

class SelectedSymptomsQuery(BaseModel):
    selected_symptoms: list

@app.post("/find_disease")
def find_disease(query: SelectedSymptomsQuery):
    selected_symptoms = query.selected_symptoms
    # Filter diseases that match at least one of the selected symptoms
    matching_diseases = df[df['Symptoms'].apply(lambda x: any(s in x for s in selected_symptoms))]

    # Sort diseases by the number of matching symptoms in descending order
    matching_diseases['match_count'] = matching_diseases['Symptoms'].apply(lambda x: sum(s in selected_symptoms for s in x))
    matching_diseases = matching_diseases.sort_values(by='match_count', ascending=False)

    # Create a list of disease information
    disease_list = []
    max_match_count_disease = None
    max_match_count = -1

    for _, row in matching_diseases.iterrows():
        disease_info = {
            'Disease': row['Name'],
            'Symptoms': row['Symptoms'],
            'Treatments': row['Treatments'],
            'MatchCount': row['match_count']
        }
        disease_list.append(disease_info)

        # Check if this disease has the maximum match count
        if row['match_count'] > max_match_count:
            max_match_count = row['match_count']
            max_match_count_disease = disease_info

    return {"disease_list": disease_list, "max_match_count_disease": max_match_count_disease}
class DiseaseListQuery(BaseModel):
    disease_list: list

class DiseaseDetail(BaseModel):
    Disease: str
    Symptoms: list
    Treatments: str
    MatchCount: int

@app.post("/pass2llm")
def pass2llm(query: DiseaseDetail):
    # Prepare the data to be sent to the LLM API
    disease_list_details = query

    # Make the API request to the Ngrok endpoint to get the public URL
    headers = {
        "Authorization": "Bearer 2npJaJjnLBj1RGPcGf0QiyAAJHJ_5qqtw2divkpoAipqN9WLG",
        "Ngrok-Version": "2"
    }
    response = requests.get("https://api.ngrok.com/endpoints", headers=headers)

    # Check if the request was successful
    if response.status_code == 200:
        llm_api_response = response.json()
        public_url = llm_api_response['endpoints'][0]['public_url']

        # Prepare the prompt with the disease list details
        prompt = f"Here is a list of diseases and their details: {disease_list_details}. Please generate a summary."

        # Make the request to the LLM API
        llm_headers = {
            "Content-Type": "application/json"
        }
        llm_payload = {
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        }
        llm_response = requests.post(f"{public_url}/api/generate", headers=llm_headers, json=llm_payload)

        # Check if the request to the LLM API was successful
        if llm_response.status_code == 200:
            llm_response_json = llm_response.json()
            return {"message": "Successfully passed to LLM!", "llm_response": llm_response_json.get("response")}
        else:
            return {"message": "Failed to get response from LLM!", "error": llm_response.text}
    else:
        return {"message": "Failed to get public URL from Ngrok!", "error": response.text}
# To run the FastAPI app with Uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
