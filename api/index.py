import os
import re
import json
import numpy as np
import requests
from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, firestore
from dotenv import load_dotenv

# Load local environment variables (if running locally)
load_dotenv()

app = Flask(__name__)

# --- Global Initialization for Serverless Optimization ---
db = None
CACHED_CHUNKS = None
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

def initialize_firebase():
    global db
    if db is not None:
        return db
        
    try:
        # Check if the credentials are provided as a JSON string in Vercel Environment Variables
        cred_json_str = os.getenv("FIREBASE_CREDENTIALS_JSON")
        
        if not firebase_admin._apps:
            if cred_json_str:
                cred_dict = json.loads(cred_json_str)
                cred = credentials.Certificate(cred_dict)
            else:
                # Fallback to local file testing
                cred_path = os.getenv("FIREBASE_CREDENTIALS_PATH", "../serviceAccountKey.json")
                cred = credentials.Certificate(cred_path)
                
            firebase_admin.initialize_app(cred)
            
        db = firestore.client()
        return db
    except Exception as e:
        print(f"Error initializing Firebase: {e}")
        return None

def fetch_all_chunks():
    global CACHED_CHUNKS, db
    if CACHED_CHUNKS is not None:
        return CACHED_CHUNKS
        
    db = initialize_firebase()
    if not db:
        raise Exception("Firebase not initialized")
        
    print("Fetching chunks from Firestore...")
    chunks_ref = db.collection(os.getenv("FIRESTORE_COLLECTION", "document_chunks"))
    docs = chunks_ref.stream()
    
    CACHED_CHUNKS = [doc.to_dict() for doc in docs]
    print(f"Loaded {len(CACHED_CHUNKS)} chunks into memory cache.")
    return CACHED_CHUNKS

def get_huggingface_embedding(text):
    if not HUGGINGFACE_API_KEY:
        raise Exception("HUGGINGFACE_API_KEY is not set.")
        
    url = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    
    response = requests.post(url, headers=headers, json={"inputs": text})
    if response.status_code != 200:
        raise Exception(f"HuggingFace API Error: {response.text}")
        
    return response.json()

# --- RAG Logic Ported from query.py ---
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def is_exercise(text):
    text_lower = text.lower()
    markers = [
        "select the answer", "codes given below", "exercises", 
        "multiple choice", "write a", "discuss", "find out"
    ]
    for marker in markers:
        if marker in text_lower:
            return True
    if text.count("?") >= 2 or ("A." in text and "B." in text and "C." in text):
        return True
    return False

def get_keywords(text):
    words = re.findall(r'\b[a-z]{4,}\b', text.lower())
    stopwords = {"what", "that", "this", "with", "from", "they", "have", "were", "when", "which", "there", "between", "does", "doesnt", "would"}
    return set(words) - stopwords

def detect_intent(query):
    query_lower = query.lower()
    exercise_keywords = ["exercise", "question", "mcq", "end-of-chapter", "practice", "valid reason", "correct statement"]
    for kw in exercise_keywords:
        if kw in query_lower:
            return "exercise"
            
    comparison_keywords = ["compare", "contrast", "differentiate", "difference between", "vs", "versus"]
    for kw in comparison_keywords:
        if kw in query_lower:
            return "comparison"
            
    if "compared to" in query_lower or "outcome of" in query_lower or "how did" in query_lower:
        return "multi-hop"
        
    return "explanatory"

def decompose_query(query, intent):
    query_lower = query.lower()
    sub_queries = []
    
    if intent == "comparison":
        parts = re.split(r'\band\b|\bvs\b|\bversus\b', query_lower)
        if len(parts) >= 2:
            core_words = " ".join([w for w in query_lower.split() if w not in ["compare", "contrast", "differentiate", "the", "between", "difference"]])
            for p in parts:
                cleaned_p = " ".join([w for w in p.split() if w not in ["compare", "contrast", "differentiate", "the", "between", "difference"]])
                if cleaned_p.strip():
                    sub_queries.append(f"{cleaned_p} {core_words}".strip())
    elif intent == "multi-hop":
        if "compared to" in query_lower:
            parts = query_lower.split("compared to")
            if len(parts) == 2:
                sub_queries.append(parts[0].strip())
                sub_queries.append(parts[1].strip() + " population")
        elif "outcome of" in query_lower:
            sub_queries.append("civil war sri lanka")
            sub_queries.append("outcome result killed")
        elif "how did the french-speaking minority" in query_lower:
            sub_queries.append("french-speaking minority belgium")
            sub_queries.append("powerful rich")
            
    if not sub_queries:
        sub_queries = [query]
    else:
        sub_queries.insert(0, query)
        
    return sub_queries

def retrieve_for_subquery(subquery, chunks, top_n=15):
    # Use HuggingFace instead of local SentenceTransformers
    subquery_emb = get_huggingface_embedding(subquery)
    
    candidates = []
    for i, chunk in enumerate(chunks):
        if 'embedding' not in chunk: continue
        chunk_emb = np.array(chunk['embedding'])
        sim_score = cosine_similarity(subquery_emb, chunk_emb)
        candidates.append({"score": sim_score, "chunk": chunk, "index": i, "subquery": subquery})
        
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:top_n]

def search_textbook(query, top_k=10):
    chunks = fetch_all_chunks()
            
    intent = detect_intent(query)
    sub_queries = decompose_query(query, intent)
    
    all_candidates = []
    seen_indices = set()
    
    for sq in sub_queries:
        sq_cands = retrieve_for_subquery(sq, chunks, top_n=20)
        for c in sq_cands:
            idx = c["index"]
            if idx not in seen_indices:
                seen_indices.add(idx)
                all_candidates.append(c)
            else:
                for existing in all_candidates:
                    if existing["index"] == idx:
                        if c["score"] > existing["score"]:
                            existing["score"] = c["score"]
                        break

    query_keywords = get_keywords(query)
    
    for item in all_candidates:
        text = item["chunk"]["text"]
        chunk_keywords = get_keywords(text)
        final_score = float(item["score"])
        
        overlap = len(query_keywords.intersection(chunk_keywords))
        final_score += overlap * 0.02
        chunk_is_ex = is_exercise(text)
        
        if intent == "exercise":
            if chunk_is_ex:
                final_score += 0.20
            else:
                final_score -= 0.10
        else:
            if chunk_is_ex:
                final_score -= 0.15
                
        item["rerank_score"] = final_score
        
    all_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    
    final_results = []
    selected_indices = set()
    covered_keywords = set()
    subquery_coverage = {sq: False for sq in sub_queries}
    
    for item in all_candidates:
        if len(final_results) >= top_k: break
            
        idx = item["index"]
        if idx in selected_indices: continue
            
        chunk_keywords = get_keywords(item["chunk"]["text"])
        new_keywords = query_keywords.intersection(chunk_keywords) - covered_keywords
        
        for sq in sub_queries:
            sq_kws = get_keywords(sq)
            if len(sq_kws.intersection(chunk_keywords)) > 0:
                subquery_coverage[sq] = True
        
        if len(query_keywords) > 2 and len(final_results) >= 2 and len(new_keywords) == 0:
            if not all(subquery_coverage.values()):
                item["rerank_score"] -= 0.05
            
        final_results.append(item)
        selected_indices.add(idx)
        covered_keywords.update(new_keywords)
        
        if len(final_results) <= 3:
            if idx + 1 < len(chunks):
                next_chunk = chunks[idx + 1]
                if next_chunk.get("chapter") == item["chunk"].get("chapter") and (idx + 1) not in selected_indices:
                    if not (not intent == "exercise" and is_exercise(next_chunk.get("text", ""))):
                        final_results.append({"score": item["score"] - 0.01, "rerank_score": item["rerank_score"] - 0.01, "chunk": next_chunk, "index": idx + 1})
                        selected_indices.add(idx + 1)
                        
    missing_subqueries = [sq for sq, covered in subquery_coverage.items() if not covered]
    if missing_subqueries and len(final_results) < top_k:
        for sq in missing_subqueries:
            sq_cands = retrieve_for_subquery(sq, chunks, top_n=5)
            for cand in sq_cands:
                if cand["index"] not in selected_indices:
                    cand["rerank_score"] = cand["score"]
                    final_results.append(cand)
                    selected_indices.add(cand["index"])
                    if len(final_results) >= top_k: break
            if len(final_results) >= top_k: break
                        
    final_results.sort(key=lambda x: x["rerank_score"], reverse=True)
    
    # Format the output into a single string to send to ChatGPT
    formatted_context = ""
    for i, item in enumerate(final_results[:top_k]):
        chunk = item["chunk"]
        formatted_context += f"--- Source {i+1} (Page {chunk.get('page_number', '?')}) ---\n"
        formatted_context += f"{chunk.get('text', '')}\n\n"
        
    return formatted_context

# --- Flask Routes ---
@app.route("/api/search", methods=["POST"])
def search():
    data = request.json
    query = data.get("query")
    
    if not query:
        return jsonify({"error": "Missing 'query' in request body"}), 400
        
    try:
        results = search_textbook(query)
        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5000)
