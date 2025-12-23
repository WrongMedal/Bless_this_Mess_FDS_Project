"""
Document Clustering and Search System Demo
Installation: pip install streamlit sentence-transformers networkx scikit-learn pymupdf transformers
Run: streamlit run demo_app.py
"""

# Disable warnings and TensorFlow
import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
warnings.filterwarnings('ignore')

import streamlit as st
import json
import torch
from pathlib import Path
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from networkx.algorithms.community import louvain_communities
import torch.nn as nn
import torch.nn.functional as F

# Page config
st.set_page_config(page_title="Document Clustering Demo", page_icon="📚", layout="wide")

# ============================================================================
# CACHE FUNCTIONS
# ============================================================================

def get_cache_path(folder_path):
    cache_dir = Path(".cache")
    cache_dir.mkdir(exist_ok=True)
    folder_name = Path(folder_path).name
    return cache_dir / f"{folder_name}_cache.json"

def load_cache(folder_path):
    cache_path = get_cache_path(folder_path)
    if cache_path.exists():
        with open(cache_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_cache(folder_path, cache_data):
    with open(get_cache_path(folder_path), 'w', encoding='utf-8') as f:
        json.dump(cache_data, f, ensure_ascii=False, indent=2)

def get_file_hash(file_path):
    stat = os.stat(file_path)
    return f"{stat.st_size}_{int(stat.st_mtime)}"

# ============================================================================
# SIAMESE MODEL
# ============================================================================

class SiameseDocumentClassifier(nn.Module):
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2", num_classes=3, dropout=0.1):
        super().__init__()
        self.encoder = SentenceTransformer(model_name)
        self.hidden_size = self.encoder.get_sentence_embedding_dimension()
        
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size * 4, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

@st.cache_resource
def load_siamese_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SiameseDocumentClassifier()
    model.to(device)
    model.eval()
    return model, device

@st.cache_resource
def load_llm_model():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map="auto"
    )
    model.eval()
    
    return model, tokenizer, device

# ============================================================================
# TEXT EXTRACTION
# ============================================================================

def extract_text_from_pdf(pdf_path):
    try:
        import fitz
        
        text_parts = []
        with fitz.open(pdf_path) as doc:
            for page_num, page in enumerate(doc):
                if page_num >= 100:
                    break
                text = page.get_text()
                if text:
                    text_parts.append(text)
        
        full_text = "\n\n".join(text_parts)
        return full_text if len(full_text) > 100 else ""
    except Exception as e:
        st.warning(f"PDF error {os.path.basename(pdf_path)}: {str(e)}")
        return ""

# ============================================================================
# EMBEDDING AND CLUSTERING
# ============================================================================

def get_single_embedding(text, model, device):
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        embedding_tensor = model.encoder.encode([text[:2000]], convert_to_tensor=True, show_progress_bar=False)
        embedding_list = embedding_tensor[0].cpu().numpy().tolist()
        return embedding_list

def create_embeddings(docs, model, device):
    embeddings = []
    progress = st.progress(0)
    
    for i, doc in enumerate(docs):
        emb = get_single_embedding(doc["full_text"], model, device)
        embeddings.append(emb)
        doc["embedding"] = emb
        progress.progress((i + 1) / len(docs))
    
    progress.empty()
    return embeddings

def build_graph_and_cluster(embeddings, doc_ids, min_sim=0.25, top_k=6, resolution=1.0):
    emb_matrix = np.vstack(embeddings)
    sim_matrix = cosine_similarity(emb_matrix)
    
    G = nx.Graph()
    for doc_id in doc_ids:
        G.add_node(doc_id)
    
    for i in range(len(doc_ids)):
        sims = sim_matrix[i].copy()
        sims[i] = -1
        top_indices = np.argsort(sims)[::-1][:top_k]
        
        for j in top_indices:
            if sims[j] >= min_sim and not G.has_edge(doc_ids[i], doc_ids[j]):
                G.add_edge(doc_ids[i], doc_ids[j], weight=float(sims[j]))
    
    communities = louvain_communities(G, weight="weight", resolution=resolution, seed=42)
    partition = {node: cid for cid, nodes in enumerate(communities) for node in nodes}
    
    clusters = {}
    for node, cid in partition.items():
        clusters.setdefault(cid, []).append(node)
    
    return G, partition, clusters

# ============================================================================
# LLM NAME GENERATION
# ============================================================================

def generate_title(text, llm_model, llm_tokenizer, device):
    snippet = text[:800]
    messages = [
        {"role": "system", "content": "Create concise titles. Output ONLY the title."},
        {"role": "user", "content": f"Create a title (max 10 words):\n\n{snippet}\n\nTitle:"}
    ]
    
    prompt = llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = llm_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    
    with torch.no_grad():
        output = llm_model.generate(**inputs, max_new_tokens=24, do_sample=False, num_beams=4)
    
    title = llm_tokenizer.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    title = title.split("\n")[0].strip('"\'')
    title = "".join(c for c in title if c.isalnum() or c in " -_.,").strip()
    
    return title if len(title) >= 3 else "Untitled"

def generate_cluster_name(titles_text, llm_model, llm_tokenizer, device):
    messages = [
        {"role": "system", "content": "Create folder names. Output ONLY the name."},
        {"role": "user", "content": f"Create a folder name (max 6 words, use hyphens, lowercase) for:\n\n{titles_text[:500]}\n\nFolder name:"}
    ]
    
    prompt = llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = llm_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    
    with torch.no_grad():
        output = llm_model.generate(**inputs, max_new_tokens=20, do_sample=False)
    
    name = llm_tokenizer.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    name = name.split("\n")[0].strip('"\'').replace(" ", "-").lower()
    name = "".join(c for c in name if c.isalnum() or c in "-_").strip("-_")[:60]
    
    return name if len(name) >= 3 else "documents"

# ============================================================================
# SEARCH
# ============================================================================

def search_similar(query, documents, model, device, top_k=5):
    query_emb = get_single_embedding(query, model, device)
    doc_embeddings = np.array([doc["embedding"] for doc in documents])
    similarities = cosine_similarity([query_emb], doc_embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    return [{"document": documents[idx], "similarity": float(similarities[idx])} for idx in top_indices]

# ============================================================================
# STREAMLIT UI
# ============================================================================

st.title("📚 Document Clustering & Search")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📁 Clustering", "🏷️ Naming", "🔍 Search"])

# TAB 1: CLUSTERING
with tab1:
    st.header("Document Clustering")
    
    with st.sidebar:
        st.header("⚙️ Parameters")
        min_similarity = st.slider("Min similarity", 0.0, 0.9, 0.25, 0.05)
        top_k = st.slider("Links per doc", 3, 15, 6)
        resolution = st.slider("Resolution", 0.3, 1.5, 1.0, 0.1)
    
    folder_path = st.text_input("PDF folder:", placeholder="/path/to/pdfs")
    
    if folder_path and os.path.isdir(folder_path):
        pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
        cache = load_cache(folder_path)
        
        col1, col2, col3 = st.columns([2, 2, 1])
        col1.success(f"✅ {len(pdf_files)} PDFs found")
        col2.info(f"💾 {len(cache)} cached")
        
        if col3.button("🗑️ Clear"):
            get_cache_path(folder_path).unlink(missing_ok=True)
            st.rerun()
        
        # Extract text
        if st.button("📄 Extract Text", type="primary"):
            cache = load_cache(folder_path)
            documents = []
            progress = st.progress(0)
            
            for i, pdf_file in enumerate(pdf_files):
                pdf_path = os.path.join(folder_path, pdf_file)
                doc_id = os.path.splitext(pdf_file)[0]
                file_hash = get_file_hash(pdf_path)
                
                if doc_id in cache and cache[doc_id].get('hash') == file_hash:
                    text = cache[doc_id].get('text', '')
                else:
                    text = extract_text_from_pdf(pdf_path)
                    cache[doc_id] = {'text': text, 'hash': file_hash}
                
                if text and len(text) > 100:
                    documents.append({"id": doc_id, "full_text": text})
                
                progress.progress((i + 1) / len(pdf_files))
            
            save_cache(folder_path, cache)
            progress.empty()
            
            st.session_state['documents'] = documents
            st.session_state['folder'] = folder_path
            st.success(f"✅ {len(documents)} documents extracted")
        
        # Clustering
        if 'documents' in st.session_state:
            st.markdown("---")
            docs = st.session_state['documents']
            st.info(f"📊 {len(docs)} documents ready")
            
            if st.button("🚀 Cluster", type="primary"):
                if len(docs) < 2:
                    st.error("Need at least 2 documents")
                else:
                    siamese_model, device = load_siamese_model()
                    
                    st.text("Creating embeddings...")
                    doc_ids = [d["id"] for d in docs]
                    embeddings = create_embeddings(docs, siamese_model, device)
                    
                    st.text("Clustering...")
                    G, partition, clusters = build_graph_and_cluster(
                        embeddings, doc_ids, min_similarity, top_k, resolution
                    )
                    
                    st.session_state['clusters'] = clusters
                    st.session_state['graph'] = G
                    
                    st.markdown("---")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Documents", len(docs))
                    col2.metric("Clusters", len(clusters))
                    col3.metric("Links", len(G.edges()))
                    
                    for cid, doc_ids in sorted(clusters.items()):
                        with st.expander(f"Cluster {cid + 1} ({len(doc_ids)} docs)"):
                            for did in doc_ids:
                                st.markdown(f"- `{did}`")

# TAB 2: NAMING
with tab2:
    st.header("Name Generation")
    
    if 'documents' not in st.session_state or 'clusters' not in st.session_state:
        st.warning("⚠️ Run clustering first")
    else:
        if st.button("✍️ Generate Names", type="primary"):
            llm_model, llm_tokenizer, llm_device = load_llm_model()
            docs = st.session_state['documents']
            clusters = st.session_state['clusters']
            
            cluster_info = {}
            doc_titles = {}
            
            # Document titles
            st.text("Generating titles...")
            progress = st.progress(0)
            for i, doc in enumerate(docs):
                title = generate_title(doc["full_text"], llm_model, llm_tokenizer, llm_device)
                doc["title"] = title
                doc_titles[doc["id"]] = title
                progress.progress((i + 1) / len(docs))
            progress.empty()
            
            # Cluster names
            st.text("Generating cluster names...")
            progress = st.progress(0)
            for i, (cid, doc_ids) in enumerate(clusters.items()):
                titles_text = " ".join([doc_titles.get(did, did) for did in doc_ids])
                name = generate_cluster_name(titles_text, llm_model, llm_tokenizer, llm_device)
                cluster_info[cid] = {"name": name, "doc_ids": doc_ids}
                progress.progress((i + 1) / len(clusters))
            progress.empty()
            
            st.session_state['cluster_info'] = cluster_info
            st.session_state['doc_titles'] = doc_titles
            
            # Show results
            st.markdown("---")
            for cid, info in sorted(cluster_info.items()):
                with st.expander(f"📁 {info['name']} ({len(info['doc_ids'])} docs)"):
                    for did in info['doc_ids']:
                        st.markdown(f"- **{doc_titles[did]}** (`{did}`)")

# TAB 3: SEARCH
with tab3:
    st.header("Search")
    
    if 'documents' not in st.session_state:
        st.warning("⚠️ Run clustering first")
    else:
        query = st.text_area("Query:", placeholder="What are you looking for?")
        num_results = st.slider("Results", 1, 10, 5)
        
        if st.button("🔍 Search", type="primary") and query:
            siamese_model, device = load_siamese_model()
            docs = st.session_state['documents']
            doc_titles = st.session_state.get('doc_titles', {})
            
            results = search_similar(query, docs, siamese_model, device, num_results)
            
            for i, res in enumerate(results, 1):
                doc = res["document"]
                sim = res["similarity"]
                title = doc_titles.get(doc['id'], doc['id'])
                
                with st.expander(f"#{i} - {title} ({sim:.3f})"):
                    st.markdown(f"**ID:** {doc['id']}")
                    st.markdown(f"**Similarity:** {sim:.3f}")
                    st.markdown(f"**Preview:** {doc['full_text'][:300]}...")

st.markdown("---")
st.markdown("*Demo v5.0 *")