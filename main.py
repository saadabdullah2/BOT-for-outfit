from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime
import uuid
import os

# Initialize FastAPI
app = FastAPI(title="AI Recommendation Engine")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Sentence Transformer (FREE, runs locally)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Sentence Transformer embedding function
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Initialize ChromaDB with PERSISTENT storage
# Create directory if it doesn't exist
PERSIST_DIR = "./chroma_db"
os.makedirs(PERSIST_DIR, exist_ok=True)

# Use PersistentClient for automatic persistence
chroma_client = chromadb.PersistentClient(
    path=PERSIST_DIR,
    settings=Settings(
        anonymized_telemetry=False,
        allow_reset=True
    )
)

print(f" ChromaDB initialized with persistent storage at: {PERSIST_DIR}")

# Pydantic Models
class ScrapeRequest(BaseModel):
    url: str
    collection_name: str = "products"
    max_pages: int = 20

class QuizResponse(BaseModel):
    answers: Dict[str, str]
    collection_name: str = "products"
    top_k: int = 10

class SearchRequest(BaseModel):
    query: str
    collection_name: str = "products"
    top_k: int = 10
    filters: Optional[Dict] = None

class Product(BaseModel):
    id: str
    name: str
    description: str
    price: Optional[str]
    image_url: Optional[str]
    url: Optional[str]
    metadata: Optional[Dict]

# Helper Functions
def scrape_beyondeast_shopify(base_url: str, max_pages: int = 20) -> List[Dict]:
    """Scrape Beyond East using Shopify's JSON API"""
    products = []
    
    try:
        for page in range(1, max_pages + 1):
            api_url = f"{base_url.rstrip('/')}/products.json?limit=250&page={page}"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json'
            }
            
            response = requests.get(api_url, headers=headers, timeout=15)
            
            if response.status_code != 200:
                break
                
            data = response.json()
            
            if not data.get('products'):
                break
            
            for item in data['products']:
                variant = item['variants'][0] if item['variants'] else {}
                
                description_parts = []
                if item.get('body_html'):
                    soup = BeautifulSoup(item['body_html'], 'html.parser')
                    description_parts.append(soup.get_text(strip=True))
                
                if item.get('tags'):
                    description_parts.append(f"Tags: {', '.join(item['tags'])}")
                
                if item.get('product_type'):
                    description_parts.append(f"Type: {item['product_type']}")
                
                if item.get('vendor'):
                    description_parts.append(f"Brand: {item['vendor']}")
                
                description = ' | '.join(filter(None, description_parts))
                
                image_url = ''
                if item.get('images') and len(item['images']) > 0:
                    image_url = item['images'][0].get('src', '')
                elif item.get('image'):
                    image_url = item['image'].get('src', '')
                
                product = {
                    'id': str(item['id']),
                    'name': item.get('title', ''),
                    'description': description,
                    'price': f"PKR {variant.get('price', 'N/A')}",
                    'compare_at_price': variant.get('compare_at_price'),
                    'image_url': image_url,
                    'url': f"{base_url}/products/{item['handle']}",
                    'product_type': item.get('product_type', ''),
                    'vendor': item.get('vendor', ''),
                    'tags': ','.join(item.get('tags', [])),
                    'available': str(variant.get('available', True)),
                    'scraped_at': datetime.now().isoformat()
                }
                
                products.append(product)
            
            if len(data['products']) < 250:
                break
        
        return products
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Scraping error: {str(e)}")

def scrape_website(url: str, max_pages: int = 20) -> List[Dict]:
    """Main scraping function"""
    if 'beyondeast.com' in url or 'shopify' in url:
        return scrape_beyondeast_shopify(url, max_pages)
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        products = []
        
        product_elements = soup.find_all(['div', 'article'], class_=lambda x: x and any(
            term in str(x).lower() for term in ['product', 'item', 'card']
        ))
        
        for elem in product_elements[:50]:
            product = {
                'id': str(uuid.uuid4()),
                'name': '',
                'description': '',
                'price': '',
                'image_url': '',
                'url': url,
                'scraped_at': datetime.now().isoformat()
            }
            
            title = elem.find(['h1', 'h2', 'h3', 'h4', 'a'])
            if title:
                product['name'] = title.get_text(strip=True)
            
            desc = elem.find(['p', 'div'])
            if desc:
                product['description'] = desc.get_text(strip=True)
            
            price = elem.find(['span', 'div', 'p'], class_=lambda x: x and 'price' in str(x).lower())
            if price:
                product['price'] = price.get_text(strip=True)
            
            img = elem.find('img')
            if img and img.get('src'):
                product['image_url'] = img.get('src')
            
            if product['name'] or product['description']:
                products.append(product)
        
        return products
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Scraping error: {str(e)}")

def get_or_create_collection(collection_name: str):
    """Get or create ChromaDB collection with Sentence Transformer embeddings"""
    try:
        collection = chroma_client.get_collection(
            name=collection_name,
            embedding_function=sentence_transformer_ef
        )
        print(f" Loaded existing collection: {collection_name} with {collection.count()} items")
    except Exception as e:
        print(f" Creating new collection: {collection_name}")
        collection = chroma_client.create_collection(
            name=collection_name,
            embedding_function=sentence_transformer_ef,
            metadata={"hnsw:space": "cosine"}
        )
    return collection

def store_in_chromadb(products: List[Dict], collection_name: str):
    """Store products with embeddings in ChromaDB"""
    collection = get_or_create_collection(collection_name)
    
    ids = []
    documents = []
    metadatas = []
    
    for product in products:
        text_parts = [
            product.get('name', ''),
            product.get('description', ''),
            product.get('product_type', ''),
            product.get('tags', ''),
            product.get('price', '')
        ]
        text = ' '.join(filter(None, text_parts))
        
        if text.strip():
            ids.append(product['id'])
            documents.append(text)
            
            # Prepare metadata (ChromaDB requires simple types)
            metadata = {
                'name': product.get('name', '')[:200],
                'description': product.get('description', '')[:500],
                'price': product.get('price', ''),
                'image_url': product.get('image_url', ''),
                'url': product.get('url', ''),
                'product_type': product.get('product_type', ''),
                'vendor': product.get('vendor', ''),
                'tags': product.get('tags', ''),
                'available': product.get('available', 'true'),
                'scraped_at': product.get('scraped_at', '')
            }
            metadatas.append(metadata)
    
    if ids:
        # Add in batches
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_docs = documents[i:i + batch_size]
            batch_meta = metadatas[i:i + batch_size]
            
            collection.add(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_meta
            )
        
        print(f" Stored {len(ids)} products in collection: {collection_name}")
    
    return len(ids)

def analyze_quiz_simple(answers: Dict[str, str]) -> str:
    """Simple quiz analysis without LLM"""
    query_parts = []
    
    for key, value in answers.items():
        if value:
            query_parts.append(f"{key}: {value}")
    
    return ' '.join(query_parts)

def search_recommendations(query: str, collection_name: str, top_k: int = 10, filters: Dict = None) -> List[Dict]:
    """Search for recommendations using semantic search"""
    try:
        collection = chroma_client.get_collection(
            name=collection_name,
            embedding_function=sentence_transformer_ef
        )
        print(f" Searching in collection: {collection_name} ({collection.count()} items)")
    except Exception as e:
        print(f" Collection not found: {collection_name}")
        return []
    
    # Prepare where filter if provided
    where_filter = None
    if filters:
        where_conditions = []
        for key, value in filters.items():
            if value:
                where_conditions.append({key: {"$eq": value}})
        
        if where_conditions:
            where_filter = {"$and": where_conditions} if len(where_conditions) > 1 else where_conditions[0]
    
    # Search in ChromaDB
    search_params = {
        "query_texts": [query],
        "n_results": top_k
    }
    
    if where_filter:
        search_params["where"] = where_filter
    
    results = collection.query(**search_params)
    
    recommendations = []
    if results['ids'] and results['ids'][0]:
        for i in range(len(results['ids'][0])):
            rec = {
                'id': results['ids'][0][i],
                'name': results['metadatas'][0][i].get('name', ''),
                'description': results['metadatas'][0][i].get('description', ''),
                'price': results['metadatas'][0][i].get('price', ''),
                'image_url': results['metadatas'][0][i].get('image_url', ''),
                'url': results['metadatas'][0][i].get('url', ''),
                'product_type': results['metadatas'][0][i].get('product_type', ''),
                'tags': results['metadatas'][0][i].get('tags', ''),
                'available': results['metadatas'][0][i].get('available', 'true'),
                'similarity_score': float(1 - results['distances'][0][i]) if results['distances'][0] else 0
            }
            recommendations.append(rec)
    
    print(f" Found {len(recommendations)} recommendations")
    return recommendations

# API Endpoints
@app.get("/")
def root():
    # Get collection stats
    collections = chroma_client.list_collections()
    stats = {}
    for col in collections:
        try:
            collection = chroma_client.get_collection(col.name)
            stats[col.name] = collection.count()
        except:
            stats[col.name] = 0
    
    return {
        "message": "AI Recommendation Engine API (Sentence Transformers - FREE)",
        "version": "2.1 - Persistent Storage",
        "model": "all-MiniLM-L6-v2",
        "persist_directory": PERSIST_DIR,
        "collections": stats,
        "endpoints": {
            "scrape": "/scrape - Scrape and store products",
            "quiz": "/quiz - Get recommendations from quiz",
            "search": "/search - Direct search",
            "collections": "/collections - List collections",
            "stats": "/stats/{collection_name} - Collection stats"
        }
    }

@app.post("/scrape")
async def scrape_and_store(request: ScrapeRequest):
    """Scrape website and store in ChromaDB"""
    try:
        products = scrape_website(request.url, request.max_pages)
        
        if not products:
            return {"message": "No products found", "count": 0}
        
        count = store_in_chromadb(products, request.collection_name)
        
        return {
            "message": "Successfully scraped and stored products",
            "count": count,
            "collection": request.collection_name,
            "sample_products": products[:3]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/quiz")
async def process_quiz(request: QuizResponse):
    """Process quiz responses and return recommendations"""
    try:
        search_query = analyze_quiz_simple(request.answers)
        
        recommendations = search_recommendations(
            search_query,
            request.collection_name,
            request.top_k
        )
        
        return {
            "analyzed_query": search_query,
            "recommendations": recommendations,
            "count": len(recommendations)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def search(request: SearchRequest):
    """Direct search in the database"""
    try:
        recommendations = search_recommendations(
            request.query,
            request.collection_name,
            request.top_k,
            request.filters
        )
        
        return {
            "query": request.query,
            "recommendations": recommendations,
            "count": len(recommendations)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/collections")
async def list_collections():
    """List all available collections"""
    try:
        collections = chroma_client.list_collections()
        collection_info = []
        
        for col in collections:
            try:
                collection = chroma_client.get_collection(col.name)
                collection_info.append({
                    "name": col.name,
                    "count": collection.count()
                })
            except:
                collection_info.append({
                    "name": col.name,
                    "count": 0
                })
        
        return {
            "collections": collection_info,
            "total": len(collections)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats/{collection_name}")
async def collection_stats(collection_name: str):
    """Get collection statistics"""
    try:
        collection = chroma_client.get_collection(
            name=collection_name,
            embedding_function=sentence_transformer_ef
        )
        count = collection.count()
        
        # Get sample data
        sample = collection.peek(limit=5)
        
        return {
            "collection": collection_name,
            "total_products": count,
            "persist_directory": PERSIST_DIR,
            "sample_products": [
                {
                    "name": sample['metadatas'][i].get('name', ''),
                    "price": sample['metadatas'][i].get('price', ''),
                    "type": sample['metadatas'][i].get('product_type', '')
                }
                for i in range(min(5, len(sample['ids'])))
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/collections/{collection_name}")
async def delete_collection(collection_name: str):
    """Delete a collection"""
    try:
        chroma_client.delete_collection(name=collection_name)
        return {"message": f"Collection '{collection_name}' deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Startup event to show existing data
@app.on_event("startup")
async def startup_event():
    print("**** AI Recommendation Engine Started ****")
    print(f" Data Directory: {PERSIST_DIR}")
    
    try:
        collections = chroma_client.list_collections()
        if collections:
            print(f"\n Found {len(collections)} existing collection(s):")
            for col in collections:
                try:
                    collection = chroma_client.get_collection(col.name)
                    print(f"   • {col.name}: {collection.count()} products")
                except:
                    print(f"   • {col.name}: Error loading")
        else:
            print("\n No existing collections found. Use /scrape to add data.")
    except Exception as e:
        print(f"\n  Error checking collections: {e}")
    
    print("="*50 + "\n")

# Run with: uvicorn main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)