"""
Financial QA System - Standalone Python Module
RAG-based approach for financial question answering.
"""

import warnings
warnings.filterwarnings("ignore")

import zipfile
import os
import pandas as pd
import torch
import time
import re
import math
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import chromadb
import numpy as np
import nltk

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class FinancialQASystem:
    """
    Financial QA System using RAG approach
    """
    
    def __init__(self, data_path='./data/gehc-annual-report-2023-2024.zip'):
        self.data_path = data_path
        self.extracted_dir = './data_extracted'
        self.plain_text_dir = './plain_text'
        self.cleaned_texts = []
        
        # RAG components
        self.indexer = None
        self.retriever = None
        self.answer_generator = None
        
        # Guardrail keywords
        self.financial_keywords = [
            'value', 'sales', 'income', 'cost', 'pbo', 'apbo', 'operations',
            'financial', 'stockholders', 'change', 'difference', 'revenue', 
            'products', 'comprehensive', 'continuing', 'service'
        ]
    
    def setup_data(self):
        """Extract and preprocess financial data"""
        print("Setting up financial data...")
        
        # Create directories
        os.makedirs(self.extracted_dir, exist_ok=True)
        os.makedirs(self.plain_text_dir, exist_ok=True)
        
        # Extract ZIP file
        with zipfile.ZipFile(self.data_path, 'r') as zip_ref:
            zip_ref.extractall(self.extracted_dir)
        
        # Convert HTML to plain text
        html_files = []
        for root, _, files in os.walk(self.extracted_dir):
            for file in files:
                if file.endswith((".html", ".htm")):
                    html_files.append(os.path.join(root, file))
        
        print(f"Found {len(html_files)} HTML files.")
        
        for html_file_path in html_files:
            try:
                # Read HTML file
                try:
                    with open(html_file_path, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                except UnicodeDecodeError:
                    with open(html_file_path, 'r', encoding='latin-1') as f:
                        html_content = f.read()
                
                # Extract text using BeautifulSoup
                soup = BeautifulSoup(html_content, 'html.parser')
                plain_text = soup.get_text(separator='\n')
                
                # Save as plain text
                relative_path = os.path.relpath(html_file_path, self.extracted_dir)
                plain_text_file_path = os.path.join(self.plain_text_dir, relative_path + ".txt")
                
                os.makedirs(os.path.dirname(plain_text_file_path), exist_ok=True)
                
                with open(plain_text_file_path, 'w', encoding='utf-8') as f:
                    f.write(plain_text)
                    
            except Exception as e:
                print(f"Error processing {html_file_path}: {e}")
        
        # Load and clean text files
        self._load_and_clean_texts()
        print("Data setup complete!")
    
    def _load_and_clean_texts(self):
        """Load and clean all plain text files"""
        all_texts = []
        
        for root, _, files in os.walk(self.plain_text_dir):
            for file in files:
                if file.endswith('.txt'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            all_texts.append(content)
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
        
        # Clean texts
        self.cleaned_texts = []
        for text in all_texts:
            cleaned = self._clean_text(text)
            if len(cleaned.strip()) > 100:  # Only keep substantial texts
                self.cleaned_texts.append(cleaned)
        
        print(f"Loaded and cleaned {len(self.cleaned_texts)} documents.")
    
    def _clean_text(self, text):
        """Clean individual text document"""
        import re
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep financial terms
        text = re.sub(r'[^\w\s\-\.\,\$\%\(\)]', ' ', text)
        
        # Remove very short lines
        lines = text.split('\n')
        lines = [line.strip() for line in lines if len(line.strip()) > 10]
        
        return '\n'.join(lines)
    
    def setup_rag_system(self):
        """Initialize RAG system components"""
        print("Setting up RAG system...")
        
        if not self.cleaned_texts:
            raise ValueError("No cleaned texts available. Run setup_data() first.")
        
        # Initialize RAG indexer
        self.indexer = RAGChunkIndexer(self.cleaned_texts)
        
        # Create chunks and embeddings
        self.indexer.create_chunks()
        self.indexer.embed_chunks()
        self.indexer.build_dense_index()
        self.indexer.build_sparse_index()
        
        # Initialize retriever
        self.retriever = HybridRAGRetriever(
            collection=self.indexer.collection,
            embedding_model=self.indexer.embedding_model,
            tfidf_vectorizer=self.indexer.tfidf_vectorizer,
            tfidf_matrix=self.indexer.tfidf_matrix,
            embedded_chunks=self.indexer.embedded_chunks
        )
        
        # Load cross-encoder for reranking
        self.retriever.load_cross_encoder()
        
        # Initialize answer generator
        self.answer_generator = GPT2AnswerGenerator()
        
        print("RAG system setup complete!")
    
    def is_relevant(self, question):
        """Check if question is relevant to financial domain"""
        return any(keyword in question.lower() for keyword in self.financial_keywords)
    
    def get_rag_response(self, question):
        """Get response using RAG approach"""
        if not self.is_relevant(question):
            return {
                "Answer": "Not applicable",
                "Confidence": "1.0000",
                "Time (s)": "0.00",
                "Method": "Guardrail (Irrelevant)"
            }
        
        if not self.retriever or not self.answer_generator:
            raise ValueError("RAG system not initialized. Run setup_rag_system() first.")
        
        start_time = time.time()
        
        # Retrieve and rerank chunks
        reranked_results = self.retriever.retrieve(
            question, n_broad_dense=5, n_broad_sparse=5, k=3, return_reranked=True
        )
        top_k_chunks = self.retriever.select_top_k_chunks(reranked_results, k=3)
        
        # Generate answer
        answer = self.answer_generator.generate_answer(top_k_chunks, question, max_length=100)
        
        # Calculate confidence
        if reranked_results and len(reranked_results) > 0:
            top_score = float(reranked_results[0]['score'])
            confidence = 1 / (1 + math.exp(-top_score))
        else:
            confidence = 0.0
        
        inference_time = time.time() - start_time
        
        return {
            "Answer": answer or "No answer generated",
            "Confidence": f"{confidence:.4f}",
            "Time (s)": f"{inference_time:.2f}",
            "Method": "RAG"
        }
    
    def get_response(self, question):
        """Get response using RAG approach - main method to call"""
        return self.get_rag_response(question)


class RAGChunkIndexer:
    """Handles chunking, embedding, and indexing for RAG"""
    
    def __init__(self, cleaned_text_data):
        self.cleaned_text_data = cleaned_text_data
        self.chunk_sizes = [100, 400]
        self.chunked_data = {}
        self.embedded_chunks = {}
        self.collection = None
        self.embedding_model = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
    
    def chunk_text(self, text, chunk_size=100, overlap=20):
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks
    
    def create_chunks(self):
        for doc_id, cleaned_text in enumerate(self.cleaned_text_data):
            for size in self.chunk_sizes:
                chunks = self.chunk_text(cleaned_text, chunk_size=size)
                if f'chunks_{size}' not in self.chunked_data:
                    self.chunked_data[f'chunks_{size}'] = []
                for i, chunk in enumerate(chunks):
                    self.chunked_data[f'chunks_{size}'].append({
                        'id': f'doc_{doc_id}_chunk_{i}_size_{size}',
                        'content': chunk,
                        'metadata': {
                            'document_id': doc_id,
                            'chunk_id': i,
                            'chunk_size': size
                        }
                    })
        
        for size, chunks in self.chunked_data.items():
            print(f"Generated {len(chunks)} chunks of size {size}.")
    
    def embed_chunks(self, model_name='all-MiniLM-L6-v2'):
        try:
            self.embedding_model = SentenceTransformer(model_name)
            print(f"Sentence embedding model '{model_name}' loaded successfully.")
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            return
        
        for size, chunks in self.chunked_data.items():
            print(f"Embedding {len(chunks)} chunks of size {size.split('_')[-1]}...")
            chunks_content = [chunk['content'] for chunk in chunks]
            try:
                embeddings = self.embedding_model.encode(chunks_content, show_progress_bar=True)
                self.embedded_chunks[size] = {
                    'chunks': chunks,
                    'embeddings': embeddings
                }
            except Exception as e:
                print(f"Error during embedding: {e}")
    
    def build_dense_index(self, collection_name="financial_chunks"):
        try:
            client = chromadb.Client()
            self.collection = client.get_or_create_collection(name=collection_name)
            print(f"ChromaDB collection '{collection_name}' ready.")
        except Exception as e:
            print(f"Error with ChromaDB: {e}")
            return
        
        if 'chunks_100' in self.embedded_chunks:
            chunks_to_add = self.embedded_chunks['chunks_100']['chunks']
            embeddings_to_add = self.embedded_chunks['chunks_100']['embeddings']
            
            ids = [chunk['id'] for chunk in chunks_to_add]
            documents = [chunk['content'] for chunk in chunks_to_add]
            metadatas = [chunk['metadata'] for chunk in chunks_to_add]
            
            try:
                self.collection.add(
                    embeddings=embeddings_to_add.tolist(),
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                print(f"Added {len(ids)} chunks to ChromaDB.")
            except Exception as e:
                print(f"Error adding to ChromaDB: {e}")
    
    def build_sparse_index(self):
        if 'chunks_100' in self.embedded_chunks:
            chunks = self.embedded_chunks['chunks_100']['chunks']
            chunks_content = [chunk['content'] for chunk in chunks]
            
            self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
            try:
                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(chunks_content)
                print("TF-IDF index built successfully.")
            except Exception as e:
                print(f"Error building TF-IDF index: {e}")
                self.tfidf_matrix = None
                self.tfidf_vectorizer = None


class HybridRAGRetriever:
    """Handles retrieval and reranking"""
    
    def __init__(self, collection, embedding_model, tfidf_vectorizer, tfidf_matrix, embedded_chunks):
        self.collection = collection
        self.embedding_model = embedding_model
        self.tfidf_vectorizer = tfidf_vectorizer
        self.tfidf_matrix = tfidf_matrix
        self.embedded_chunks = embedded_chunks
        self.cross_encoder_model = None
        
        # Prepare chunks for sparse retrieval
        self.chunks_to_embed = self.embedded_chunks['chunks_100']['chunks'] if 'chunks_100' in self.embedded_chunks else []
        
        # Download stopwords
        from nltk.corpus import stopwords
        self.stop_words = set(stopwords.words('english'))
    
    def load_cross_encoder(self, model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
        try:
            self.cross_encoder_model = CrossEncoder(model_name)
            print(f"Cross-encoder model loaded.")
        except Exception as e:
            print(f"Error loading cross-encoder: {e}")
    
    def preprocess_query(self, query):
        import re
        query = query.lower()
        query = re.sub(r'[^a-z0-9\s]', '', query)
        query = ' '.join([word for word in query.split() if word not in self.stop_words])
        return query
    
    def dense_retrieve(self, query, n_results=5):
        if not self.collection or not self.embedding_model:
            return []
        
        try:
            query_embedding = self.embedding_model.encode([query]).tolist()
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=n_results,
                include=['documents', 'metadatas']
            )
            
            retrieved_chunks = []
            if results and results['ids'] and results['documents']:
                for i in range(len(results['ids'][0])):
                    retrieved_chunks.append({
                        'id': results['ids'][0][i],
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i]
                    })
            
            return retrieved_chunks
        except Exception as e:
            print(f"Dense retrieval error: {e}")
            return []
    
    def sparse_retrieve_tfidf(self, query, n_results=5):
        if not self.tfidf_vectorizer or self.tfidf_matrix is None:
            return []
        
        try:
            query_tfidf = self.tfidf_vectorizer.transform([query])
            similarities = cosine_similarity(query_tfidf, self.tfidf_matrix).flatten()
            top_indices = np.argsort(similarities)[-n_results:][::-1]
            
            retrieved_chunks = []
            for idx in top_indices:
                if idx < len(self.chunks_to_embed):
                    retrieved_chunks.append(self.chunks_to_embed[idx])
            
            return retrieved_chunks
        except Exception as e:
            print(f"Sparse retrieval error: {e}")
            return []
    
    def combine_retrieval_results(self, dense_results, sparse_results):
        combined_chunks = {}
        for chunk in dense_results + sparse_results:
            combined_chunks[chunk['id']] = chunk
        return list(combined_chunks.values())
    
    def rerank(self, query, combined_results):
        if not self.cross_encoder_model or not combined_results:
            return [{'chunk': chunk, 'score': 0.5} for chunk in combined_results]
        
        try:
            sentence_pairs = [[query, chunk['content']] for chunk in combined_results]
            scores = self.cross_encoder_model.predict(sentence_pairs)
            
            scored_results = []
            for i, chunk in enumerate(combined_results):
                scored_results.append({
                    'chunk': chunk,
                    'score': scores[i]
                })
            
            return sorted(scored_results, key=lambda x: x['score'], reverse=True)
        except Exception as e:
            print(f"Reranking error: {e}")
            return [{'chunk': chunk, 'score': 0.5} for chunk in combined_results]
    
    def select_top_k_chunks(self, reranked_results, k=3):
        return [item['chunk'] for item in reranked_results[:k]]
    
    def retrieve(self, user_query, n_broad_dense=5, n_broad_sparse=5, k=3, return_reranked=False):
        preprocessed_query = self.preprocess_query(user_query)
        
        dense_results = self.dense_retrieve(preprocessed_query, n_results=n_broad_dense)
        sparse_results = self.sparse_retrieve_tfidf(preprocessed_query, n_results=n_broad_sparse)
        combined_results = self.combine_retrieval_results(dense_results, sparse_results)
        reranked_results = self.rerank(preprocessed_query, combined_results)
        
        if return_reranked:
            return reranked_results
        
        return self.select_top_k_chunks(reranked_results, k=k)


class GPT2AnswerGenerator:
    """Generates answers using GPT-2"""
    
    def __init__(self, model_name="gpt2"):
        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            self.model = GPT2LMHeadModel.from_pretrained(model_name)
            
            # Set padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            print(f"GPT-2 model loaded: {model_name}")
        except Exception as e:
            print(f"Error loading GPT-2: {e}")
            self.tokenizer = None
            self.model = None
    
    def generate_answer(self, top_k_chunks, user_query, max_length=150):
        if not self.tokenizer or not self.model or not top_k_chunks:
            return None
        
        context = "\n".join([chunk['content'] for chunk in top_k_chunks])
        prompt = f"Context:\n{context}\n\nQuestion: {user_query}\n\nAnswer:"
        
        max_prompt_length = self.tokenizer.model_max_length - max_length
        encoded_prompt = self.tokenizer.encode(prompt, max_length=max_prompt_length, truncation=True, return_tensors="pt")
        
        try:
            output_sequences = self.model.generate(
                encoded_prompt,
                max_length=len(encoded_prompt[0]) + max_length,
                num_return_sequences=1,
                temperature=0.3,
                top_k=40,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True
            )
            
            generated_text = self.tokenizer.decode(output_sequences[0], skip_special_tokens=True)
            
            # Extract answer
            answer_start = generated_text.find("Answer:")
            if answer_start != -1:
                final_answer = generated_text[answer_start + len("Answer:"):].strip()
            else:
                prompt_length = len(self.tokenizer.decode(encoded_prompt[0], skip_special_tokens=True))
                final_answer = generated_text[prompt_length:].strip()
            
            # Limit to 50 words
            final_answer = ' '.join(final_answer.split()[:50])
            return final_answer
            
        except Exception as e:
            print(f"Generation error: {e}")
            return None


def main():
    """Example of how to use the Financial QA System"""
    
    # Initialize system
    qa_system = FinancialQASystem()
    
    # Setup data and models
    print("Setting up data...")
    qa_system.setup_data()
    
    print("Setting up RAG system...")
    qa_system.setup_rag_system()
    
    # Test questions
    test_questions = [
        "What was the value of sales of products in 2023?",
        "How much did net income change from 2023 to 2024?",
        "What is the capital of France?"  # Irrelevant question
    ]
    
    # Get responses
    for question in test_questions:
        result = qa_system.get_response(question)
        print("\n" + "="*80)
        print(f"Question: {question}")
        print(f"Answer: {result['Answer']}")
        print(f"Confidence: {result['Confidence']}, Time: {result['Time (s)']}s")
        print(f"Method: {result['Method']}")


if __name__ == "__main__":
    main()
