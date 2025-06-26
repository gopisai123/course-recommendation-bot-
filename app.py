import gradio as gr
import pandas as pd
import json
import re
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from transformers import pipeline, AutoTokenizer
from langchain_community.llms import HuggingFacePipeline

# Initialize embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def load_course_db():
    try:
        df = pd.read_csv("courses.csv")
        print(f"Loaded CSV with columns: {df.columns.tolist()}")
        texts = []
        for _, row in df.iterrows():
            title = row.get('title') or row.get('course_title') or row.get('name') or "Untitled Course"
            description = row.get('description') or row.get('course_description') or row.get('summary') or "No description available"
            # Truncate description more aggressively
            if len(str(description)) > 150:
                description = str(description)[:147] + "..."
            level = row.get('Level') or row.get('level') or "Not specified"
            subject = row.get('subject') or "General"
            url = row.get('url') or row.get('course_url') or "#"
            text = f"{title}: {description} | Level: {level} | URL: {url}"  # Simplified
            texts.append(text)
        return Chroma.from_texts(texts, embeddings), df
    except Exception as e:
        print(f"Error loading CSV: {str(e)}")
        sample_courses = [
            {"title": "Python Fundamentals", "description": "Learn programming", "level": "Beginner", "url": "https://example.com/python"},
            {"title": "Machine Learning", "description": "ML techniques", "level": "Intermediate", "url": "https://example.com/ml"},
        ]
        df = pd.DataFrame(sample_courses)
        texts = [f"{row['title']}: {row['description']} | Level: {row['level']} | URL: {row['url']}" for _, row in df.iterrows()]
        return Chroma.from_texts(texts, embeddings), df

vector_db, course_df = load_course_db()

# Initialize local LLM with FIXED parameters
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
generator = pipeline(
    "text-generation",
    model="distilgpt2",
    max_new_tokens=150,  # Reduced output length
    max_length=1024,     # Matches model's context window
    truncation=True,      # Essential for long inputs
    temperature=0.5
)
llm = HuggingFacePipeline(pipeline=generator)

def extract_json_from_response(response):
    """Robust JSON extraction from model response"""
    try:
        # Find first valid JSON array in response
        json_match = re.search(r'\[(\s*\{.*?\}\s*,?\s*)+\]', response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
        
        # Manual parsing as fallback
        courses = []
        entries = re.split(r'\n\d+\.|\n-|\n\*', response)
        for entry in entries:
            if not entry.strip():
                continue
            title = re.search(r'Title: (.+)', entry)
            reason = re.search(r'Reason: (.+)', entry)
            level = re.search(r'Level: (.+)', entry)
            url = re.search(r'URL: (\S+)', entry)
            
            if title and reason and level and url:
                courses.append({
                    "title": title.group(1).strip(),
                    "reason": reason.group(1).strip(),
                    "level": level.group(1).strip(),
                    "url": url.group(1).strip()
                })
        return courses if courses else [{"error": "Couldn't parse response", "raw": response}]
    except Exception:
        return [{"error": "JSON parsing failed", "raw": response}]

def recommend_courses(query):
    try:
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_db.as_retriever(search_kwargs={"k": 3})
        )
        
        # Optimized prompt
        prompt = f"""
        User background: "{query}"
        Recommend exactly 3 courses with these details:
        - Title
        - Reason: 1-sentence justification
        - Level: Difficulty
        - URL: Direct link
        
        Format as JSON only:
        [
          {{
            "title": "Course Name",
            "reason": "Brief reason",
            "level": "Difficulty",
            "url": "https://link.com"
          }},
          ... (2 more)
        ]
        """
        
        # Token length check
        tokens = tokenizer(prompt, return_tensors="pt").input_ids
        if tokens.shape[1] > 800:  # More conservative limit
            return [{"error": "Input too long. Please simplify your query"}]
            
        result = qa.invoke({"query": prompt})
        response = result["result"]
        return extract_json_from_response(response)
            
    except Exception as e:
        return [{"error": f"Recommendation failed: {str(e)}"}]

# ... (generate_learning_path and Gradio interface remain unchanged) ...
