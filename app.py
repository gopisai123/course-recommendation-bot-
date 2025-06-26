import gradio as gr
import pandas as pd
import json
import re
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline  # Updated imports
from langchain.chains import RetrievalQA
from transformers import pipeline, AutoTokenizer

# Global variables for resource caching
vector_db = None
course_df = None
llm = None
tokenizer = None

def init_resources():
    """Initialize resources only once"""
    global vector_db, course_df, llm, tokenizer
    
    if vector_db is None:
        # Initialize embedding model
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Load course database
        try:
            df = pd.read_csv("courses.csv")
            print(f"Loaded CSV with columns: {df.columns.tolist()}")
            texts = []
            for _, row in df.iterrows():
                title = row.get('title') or row.get('course_title') or "Untitled Course"
                description = row.get('description') or row.get('course_description') or "No description"
                # Truncate description
                if len(str(description)) > 150:
                    description = str(description)[:147] + "..."
                level = row.get('Level') or "Not specified"
                url = row.get('url') or row.get('course_url') or "#"
                text = f"{title}: {description} | Level: {level} | URL: {url}"
                texts.append(text)
            vector_db = Chroma.from_texts(texts, embeddings)
            course_df = df
        except Exception as e:
            print(f"Error loading CSV: {str(e)}")
            # Fallback to sample data
            sample_courses = [
                {"title": "Python Fundamentals", "description": "Learn programming", "level": "Beginner", "url": "https://example.com/python"},
                {"title": "Machine Learning", "description": "ML techniques", "level": "Intermediate", "url": "https://example.com/ml"},
            ]
            df = pd.DataFrame(sample_courses)
            texts = [f"{row['title']}: {row['description']} | Level: {row['level']} | URL: {row['url']}" for _, row in df.iterrows()]
            vector_db = Chroma.from_texts(texts, embeddings)
            course_df = df
    
    if llm is None:
        # Initialize LLM
        tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        generator = pipeline(
            "text-generation",
            model="distilgpt2",
            max_new_tokens=150,
            max_length=1024,
            truncation=True,
            temperature=0.5
        )
        llm = HuggingFacePipeline(pipeline=generator)
    
    return vector_db, course_df, llm, tokenizer

def extract_json_from_response(response):
    """Robust JSON extraction with enhanced parsing"""
    try:
        # Attempt direct JSON parsing
        json_match = re.search(r'(\[\s*\{.*?\}\s*(?:,\s*\{.*?\}\s*)*\])', response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))
        
        # Fallback: Pattern-based extraction
        courses = []
        pattern = r'"?title"?:\s*"([^"]+)".*?"reason"?:\s*"([^"]+)".*?"level"?:\s*"([^"]+)".*?"url"?:\s*"([^"]+)"'
        matches = re.finditer(pattern, response, re.DOTALL)
        for match in matches:
            courses.append({
                "title": match.group(1),
                "reason": match.group(2),
                "level": match.group(3),
                "url": match.group(4)
            })
        return courses if courses else [{"error": "Couldn't parse response", "raw": response}]
    except Exception as e:
        return [{"error": f"JSON parsing failed: {str(e)}", "raw": response}]

def recommend_courses(query):
    # Initialize resources on first call
    vector_db, course_df, llm, tokenizer = init_resources()
    
    try:
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_db.as_retriever(search_kwargs={"k": 3})
        )
        
        # Optimized prompt with JSON schema
        prompt = f"""
        User background: "{query}"
        Recommend exactly 3 courses in JSON format with these keys:
        - title (string)
        - reason (string, 1-sentence justification)
        - level (string, difficulty)
        - url (string, direct link)
        
        Output ONLY valid JSON array:
        [
          {{
            "title": "Course Name",
            "reason": "Brief reason",
            "level": "Beginner/Intermediate/Advanced",
            "url": "https://link.com"
          }},
          {{...}},
          {{...}}
        ]
        """
        
        # Token length check
        tokens = tokenizer(prompt, return_tensors="pt").input_ids
        if tokens.shape[1] > 800:
            return [{"error": "Input too long. Please simplify your query"}]
            
        result = qa.invoke({"query": prompt})
        response = result["result"]
        return extract_json_from_response(response)
            
    except Exception as e:
        return [{"error": f"Recommendation failed: {str(e)}"}]

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("## 🎓 AI Course Recommender")
    with gr.Row():
        background = gr.Textbox(label="Your Background/Goals", placeholder="e.g., '3rd year CS student interested in ML'")
        recommend_btn = gr.Button("Recommend Courses")
    
    output = gr.JSON(label="Recommended Courses")
    
    recommend_btn.click(
        fn=recommend_courses,
        inputs=background,
        outputs=output
    )

# Initialize resources before launch
init_resources()

# Launch app
if __name__ == "__main__":
    demo.launch()
