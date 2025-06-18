
import csv
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI

# -------- Embedding Model Class --------
class EmbeddingModel:
    def __init__(self):
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key="ollama",
            api_base="http://localhost:11434/v1",
            model_name="nomic-embed-text"
        )

# -------- LLM Model Class --------
class LLMMODEL:
    def __init__(self):
        self.client = OpenAI(
            base_url='http://localhost:11434/v1',
            api_key='ollama',
        )
        self.model_name = "llama3.2"

    def generate(self, messages):
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.0,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"error: {e}"

# -------- CSV Functions --------
def generate_symptom_doctor_csv_for_embeddings(file_path):
    data = [
        ("chest pain", "Cardiologist"),
        ("skin rash", "Dermatologist"),
        ("vision problems", "Ophthalmologist"),
        ("toothache", "Dentist"),
        ("back pain", "Orthopedic"),
        ("frequent urination", "Urologist"),
        ("stomach ache", "Gastroenterologist"),
        ("fever", "General Physician"),
        ("anxiety", "Psychiatrist"),
        ("pregnancy care", "Gynecologist")
    ]

    with open(file_path, mode='w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["id", "text"])
        writer.writeheader()
        for idx, (symptom, specialist) in enumerate(data, 1):
            text = f"A person experiencing {symptom} should consult a {specialist}."
            writer.writerow({"id": str(idx), "text": text})

def load_symptom_doctor_csv(file_path):
    with open(file_path, mode='r') as csvfile:
        reader = csv.DictReader(csvfile)
        return list(reader)

# -------- Chroma Setup --------
def setup_chromadb(documents, embedding_model):
    client = chromadb.Client()

    try:
        client.delete_collection("symptom-doctor")
    except:
        pass

    collection = client.create_collection(
        name="symptom-doctor",
        embedding_function=embedding_model.embedding_function
    )

    collection.add(
        documents=[doc["text"] for doc in documents],
        ids=[doc["id"] for doc in documents],
        metadatas=[{"source": "symptom-doctor"} for _ in documents]
    )

    return collection

# -------- RAG Utilities --------
def find_related_chunks(query, collection, top_k=2):
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )

    documents = results["documents"][0]
    metadata = results.get("metadatas", [[{}] * len(documents)])[0]
    distances = results.get("distances", [[{}] * len(documents)])[0]

    return list(zip(metadata, zip(documents, distances)))

def augment_prompts_with_related_chunks(query, related_chunks):
    context = "\n".join([chunk[0].get("source", "") + ": " + chunk[1][0] for chunk in related_chunks])
    augmented_prompt = [f"Context: {context}\nQuery: {query}\nAnswer:"]
    return augmented_prompt

def rag_pipeline(query, collection, llm_model, top_k=2):
    related_chunks = find_related_chunks(query, collection, top_k=top_k)
    augmented_prompt = augment_prompts_with_related_chunks(query, related_chunks)

    response = llm_model.generate([
        {
            "role": "system",
            "content": "You are a medical expert. Based on the symptoms, suggest which doctor or specialist the patient should visit. Only answer based on the document provided."
        },
        {
            "role": "user",
            "content": augmented_prompt[0],
        }
    ])

    reference = [[chunk[1][0] for chunk in related_chunks]]  # list of related documents
    return response, reference

# -------- Main --------
def main():
    print("hello")

    embedding_model = EmbeddingModel()
    llm_model = LLMMODEL()

    generate_symptom_doctor_csv_for_embeddings("symptom-doctor.csv")
    documents = load_symptom_doctor_csv("symptom-doctor.csv")
    collection = setup_chromadb(documents, embedding_model)

    query = input("please enter what you are experiencing: ")
    response, reference = rag_pipeline(query, collection, llm_model)

    print("\nResponse:\n", response)
    print("\nReference:\n", reference)

if __name__ == "__main__":
    main()
