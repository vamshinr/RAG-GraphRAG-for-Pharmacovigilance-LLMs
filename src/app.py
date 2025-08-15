from flask import Flask, render_template, request, jsonify
from rag import RAGSystem
from graphrag import GraphRAGSystem
from transformers import pipeline

# --- Configuration ---
# Replace with your Neo4j credentials
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "your_password_here" 

# --- Initialization ---
app = Flask(__name__, template_folder='../templates', static_folder='../static')

# Initialize RAG and GraphRAG systems
print("Initializing systems...")
rag_system = RAGSystem()
graph_rag_system = GraphRAGSystem(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

# Initialize a simple text generation pipeline from Hugging Face
# Using a small, efficient model for demonstration purposes
generator = pipeline('text-generation', model='EleutherAI/gpt-neo-125M')
print("All systems ready.")


def generate_response(query, retrieved_docs):
    """
    Generates a response using the LLM based on the retrieved context.
    """
    if not retrieved_docs:
        return "I couldn't find any information on that drug. Please check the spelling."

    context = "\n".join(retrieved_docs)
    prompt = f"""
    Based on the following information, answer the user's question.
    
    Information:
    {context}
    
    Question: {query}
    
    Answer:
    """
    
    # Generate response
    response = generator(prompt, max_length=150, num_return_sequences=1, pad_token_id=generator.tokenizer.eos_token_id)
    return response[0]['generated_text'].split("Answer:")[1].strip()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    user_query = data.get('query')
    method = data.get('method') # 'rag' or 'graphrag'
    
    if not user_query:
        return jsonify({'error': 'No query provided'}), 400

    retrieved_docs = []
    if method == 'rag':
        retrieved_docs = rag_system.search(user_query, k=3)
    elif method == 'graphrag':
        # For GraphRAG, we need to extract the drug name.
        # This is a simple heuristic; a more robust solution would use NLP.
        drug_name = user_query.split(" of ")[-1].replace("?", "").strip().capitalize()
        retrieved_docs = graph_rag_system.search(drug_name)
    
    response = generate_response(user_query, retrieved_docs)
    
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
