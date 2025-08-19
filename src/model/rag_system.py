from question_generator import FinancialReportProcessor
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# Usage:
zip_file_path = './data/gehc-annual-report-2023-2024.zip'
extracted_dir_path = './data/gehc_fin_extracted'
plain_text_dir_path = './data/gehc_fin_plain_text'

processor = FinancialReportProcessor(zip_file_path, extracted_dir_path, plain_text_dir_path)
processor.extract_and_convert_html_to_text()
processor.load_plain_text_files()
processor.clean_all_texts()
processor.segment_reports()
processor.extract_financial_data()
processor.generate_questions()
processor.generate_answers()
class RAGChunkIndexer:
    """
    Handles chunking, embedding, dense/sparse indexing for RAG pipeline.
    """

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
            if len(chunks) > 0:
                print(f"First chunk ({size}): {chunks[0]['content'][:200]}...")

    def embed_chunks(self, model_name='all-MiniLM-L6-v2'):
        try:
            self.embedding_model = SentenceTransformer(model_name)
            print(f"Sentence embedding model '{model_name}' loaded successfully.")
        except Exception as e:
            print(f"Error loading sentence embedding model: {e}")
            print("Please ensure you have an active internet connection to download the model.")
            self.embedding_model = None
            return

        if self.embedding_model is not None:
            for size, chunks in self.chunked_data.items():
                print(f"Embedding {len(chunks)} chunks of size {size.split('_')[-1]}...")
                chunks_content = [chunk['content'] for chunk in chunks]
                try:
                    embeddings = self.embedding_model.encode(chunks_content, show_progress_bar=True)
                    self.embedded_chunks[size] = {
                        'chunks': chunks,
                        'embeddings': embeddings
                    }
                    print(f"Finished embedding chunks of size {size.split('_')[-1]}. Shape of embeddings: {embeddings.shape}")
                except Exception as e:
                    print(f"Error during embedding for chunk size {size.split('_')[-1]}: {e}")
                    self.embedded_chunks[size] = None
        else:
            print("Embedding model not loaded, skipping embedding step.")

    def build_dense_index(self, collection_name="financial_report_chunks"):
        try:
            client = chromadb.Client()
            print("ChromaDB client initialized.")
        except Exception as e:
            print(f"Error initializing ChromaDB client: {e}")
            return

        try:
            self.collection = client.get_or_create_collection(name=collection_name)
            print(f"ChromaDB collection '{collection_name}' created or retrieved.")
        except Exception as e:
            print(f"Error getting or creating ChromaDB collection: {e}")
            self.collection = None
            return

        if self.collection is not None and self.embedded_chunks.get('chunks_100') and self.embedded_chunks['chunks_100']['embeddings'] is not None:
            chunks_to_add = self.embedded_chunks['chunks_100']['chunks']
            embeddings_to_add = self.embedded_chunks['chunks_100']['embeddings']
            ids = [chunk['id'] for chunk in chunks_to_add]
            documents = [chunk['content'] for chunk in chunks_to_add]
            metadatas = [chunk['metadata'] for chunk in chunks_to_add]
            batch_size = 100
            for i in range(0, len(ids), batch_size):
                batch_ids = ids[i:i + batch_size]
                batch_documents = documents[i:i + batch_size]
                batch_embeddings = embeddings_to_add[i:i + batch_size]
                batch_metadatas = metadatas[i:i + batch_size]
                try:
                    self.collection.add(
                        embeddings=batch_embeddings.tolist(),
                        documents=batch_documents,
                        metadatas=batch_metadatas,
                        ids=batch_ids
                    )
                    print(f"Added batch {i//batch_size + 1} to ChromaDB.")
                except Exception as e:
                    print(f"Error adding batch {i//batch_size + 1} to ChromaDB: {e}")
            print(f"Finished adding {len(ids)} chunks to ChromaDB collection '{collection_name}'.")
        if self.collection is not None:
            try:
                count = self.collection.count()
                print(f"Total items in ChromaDB collection '{collection_name}': {count}")
            except Exception as e:
                print(f"Error getting count from ChromaDB collection: {e}")

    def build_sparse_index(self):
        if 'chunks_100' in self.embedded_chunks:
            chunks_to_embed = self.embedded_chunks['chunks_100']['chunks']
            chunks_content = [chunk['content'] for chunk in chunks_to_embed]
            self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
            try:
                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(chunks_content)
                print("TF-IDF vectorizer fitted and matrix created successfully.")
                print(f"Shape of TF-IDF matrix: {self.tfidf_matrix.shape}")
            except Exception as e:
                print(f"Error creating TF-IDF matrix: {e}")
                self.tfidf_vectorizer = None
                self.tfidf_matrix = None
        else:
            print("Chunks of size 100 not found in chunked_data. Cannot build TF-IDF index.")
            self.tfidf_vectorizer = None
            self.tfidf_matrix = None

# The tfidf_matrix now represents the sparse index of our chunks.

"""### 2.3 Hybrid Retrieval Pipeline

#### 2.3.1 Preprocess data clean
"""

class HybridRAGRetriever:
    def get_user_response(self, user_query, answer_generator, n_broad_dense=10, n_broad_sparse=10, k=3):
        """
        Retrieve top-k relevant chunks for the user query and generate an answer.
        Does NOT regenerate questions/answers or rebuild indexes.
        """
        top_k_chunks = self.retrieve(user_query, n_broad_dense, n_broad_sparse, k)
        answer = answer_generator.generate_answer(top_k_chunks, user_query)
        return {
            "answer": answer,
            "chunks": top_k_chunks
        }
    """
    Implements a hybrid retrieval pipeline for RAG using dense, sparse, and cross-encoder reranking.
    """

    def __init__(self, collection, embedding_model, tfidf_vectorizer, tfidf_matrix, embedded_chunks):
        self.collection = collection
        self.embedding_model = embedding_model
        self.tfidf_vectorizer = tfidf_vectorizer
        self.tfidf_matrix = tfidf_matrix
        self.embedded_chunks = embedded_chunks
        self.cross_encoder_model = None

        # Prepare chunks for sparse retrieval
        self.chunks_to_embed = self.embedded_chunks['chunks_100']['chunks'] if 'chunks_100' in self.embedded_chunks else []

        # Download stopwords if not already downloaded
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        from nltk.corpus import stopwords
        self.stop_words = set(stopwords.words('english'))

    def preprocess_query(self, query):
        import re
        # Convert to lowercase
        query = query.lower()
        # Remove special characters and punctuation
        query = re.sub(r'[^a-z0-9\s]', '', query)
        # Remove stopwords
        query = ' '.join([word for word in query.split() if word not in self.stop_words])
        return query

    def generate_query_embedding(self, query):
        if self.embedding_model is None:
            print("Embedding model is not loaded. Cannot generate query embedding.")
            return None
        try:
            query_embedding = self.embedding_model.encode(query)
            print("Query embedding generated successfully.")
            return query_embedding
        except Exception as e:
            print(f"Error generating query embedding: {e}")
            return None

    def dense_retrieve(self, query, n_results=5):
        if self.collection is None or self.embedding_model is None:
            print("ChromaDB collection or embedding model not loaded. Cannot perform dense retrieval.")
            return []
        try:
            query_embedding = self.embedding_model.encode([query]).tolist()
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=n_results,
                include=['documents', 'metadatas']
            )
            retrieved_chunks = []
            if results and results['ids'] and results['documents'] and results['metadatas']:
                for i in range(len(results['ids'][0])):
                    retrieved_chunks.append({
                        'id': results['ids'][0][i],
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i]
                    })
            print(f"Dense retrieval found {len(retrieved_chunks)} results.")
            return retrieved_chunks
        except Exception as e:
            print(f"Error during dense retrieval: {e}")
            return []

    def sparse_retrieve_tfidf(self, query, n_results=5):
        if self.tfidf_vectorizer is None or self.tfidf_matrix is None or not self.chunks_to_embed:
            print("TF-IDF vectorizer, matrix, or chunks not available. Cannot perform sparse retrieval.")
            return []
        try:
            query_tfidf = self.tfidf_vectorizer.transform([query])
            cosine_similarities = cosine_similarity(query_tfidf, self.tfidf_matrix).flatten()
            top_n_indices = np.argpartition(cosine_similarities, -n_results)[-n_results:]
            top_n_indices = top_n_indices[top_n_indices < len(self.chunks_to_embed)]
            sorted_indices = top_n_indices[np.argsort(cosine_similarities[top_n_indices])][::-1]
            retrieved_chunks = []
            for idx in sorted_indices:
                int_idx = int(idx)
                retrieved_chunks.append({
                    'id': self.chunks_to_embed[int_idx]['id'],
                    'content': self.chunks_to_embed[int_idx]['content'],
                    'metadata': self.chunks_to_embed[int_idx]['metadata']
                })
            print(f"Sparse retrieval found {len(retrieved_chunks)} results.")
            return retrieved_chunks
        except Exception as e:
            print(f"Error during sparse retrieval: {e}")
            return []

    def combine_retrieval_results(self, dense_results, sparse_results):
        combined_chunks = {}
        for chunk in dense_results:
            combined_chunks[chunk['id']] = chunk
        for chunk in sparse_results:
            combined_chunks[chunk['id']] = chunk
        return list(combined_chunks.values())

    def load_cross_encoder(self, model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
        try:
            self.cross_encoder_model = CrossEncoder(model_name)
            print(f"Cross-encoder model '{model_name}' loaded successfully.")
        except Exception as e:
            print(f"Error loading cross-encoder model: {e}")
            self.cross_encoder_model = None
            print("Please ensure you have an active internet connection to download the model.")

    def rerank(self, query, combined_results):
        if self.cross_encoder_model is None or not combined_results:
            print("\nSkipping reranking due to missing cross-encoder model or combined results.")
            return []
        print("\nReranking combined results...")
        sentence_pairs = [[query, chunk['content']] for chunk in combined_results]
        try:
            reranking_scores = self.cross_encoder_model.predict(sentence_pairs)
            scored_results = []
            for i, chunk in enumerate(combined_results):
                scored_results.append({
                    'chunk': chunk,
                    'score': reranking_scores[i]
                })
            reranked_results = sorted(scored_results, key=lambda x: x['score'], reverse=True)
            print(f"Finished reranking. Top score: {reranked_results[0]['score'] if reranked_results else 'N/A'}")
            return reranked_results
        except Exception as e:
            print(f"Error during reranking: {e}")
            return []

    def select_top_k_chunks(self, reranked_results, k=3):
        top_k_chunks = [item['chunk'] for item in reranked_results[:k]]
        print(f"\nSelected top {k} chunks for response generation.")
        for chunk in top_k_chunks:
            print(f"- ID: {chunk['id']}, Content: {chunk['content'][:150]}...")
        return top_k_chunks

    def retrieve(self, user_query, n_broad_dense=10, n_broad_sparse=10, k=3):
        # Preprocess the query
        preprocessed_query = self.preprocess_query(user_query)
        # Broad retrieval
        broad_dense_results = self.dense_retrieve(preprocessed_query, n_results=n_broad_dense)
        broad_sparse_results = self.sparse_retrieve_tfidf(preprocessed_query, n_results=n_broad_sparse)
        # Combine
        combined_results = self.combine_retrieval_results(broad_dense_results, broad_sparse_results)
        # Rerank
        reranked_results = self.rerank(preprocessed_query, combined_results)
        # Select top-k
        top_k_chunks = self.select_top_k_chunks(reranked_results, k=k)
        return top_k_chunks

# Step 5: Generate Answer using a small generative model (GPT-2 Small)
# Install transformers library if not already installed
# %pip install transformers

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import zipfile
import os
from bs4 import BeautifulSoup

class GPT2AnswerGenerator:
    """
    Loads GPT-2 Small model and tokenizer, and generates answers given top retrieved chunks and user query.
    """

    def __init__(self, model_name="gpt2"):
        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            self.model = GPT2LMHeadModel.from_pretrained(model_name)
            print(f"\nLoaded generative model: {model_name}")
        except Exception as e:
            print(f"Error loading generative model {model_name}: {e}")
            self.tokenizer = None
            self.model = None

    def generate_answer(self, top_k_chunks, user_query):
        """
        Generate an answer using GPT-2 given the top retrieved chunks and user query.
        Returns the generated answer as a string, or None if generation fails.
        """
        if self.tokenizer is not None and self.model is not None and top_k_chunks:
            context = "\n".join([chunk['content'] for chunk in top_k_chunks])
            prompt = f"Context:\n{context}\n\nQuestion: {user_query}\n\nAnswer:"
            max_model_input_length = self.tokenizer.model_max_length
            max_prompt_length = max_model_input_length - 50
            encoded_prompt = self.tokenizer.encode(prompt, max_length=max_prompt_length, truncation=True, return_tensors="pt")
            try:
                print("\nGenerating answer...")
                output_sequences = self.model.generate(
                    encoded_prompt,
                    max_length=max_model_input_length,
                    num_return_sequences=1,
                    no_repeat_ngram_size=2,
                    early_stopping=True,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.95
                )
                generated_text = self.tokenizer.decode(output_sequences[0], skip_special_tokens=True)
                answer_start = generated_text.find("Answer:")
                if answer_start != -1:
                    final_answer = generated_text[answer_start + len("Answer:"):].strip()
                else:
                    final_answer = generated_text.strip()
                print("\nGenerated Answer:")
                print(final_answer)
                return final_answer
            except Exception as e:
                print(f"Error during answer generation: {e}")
                return None
        else:
            print("\nSkipping answer generation due to missing model, tokenizer, or chunks.")
            return None
