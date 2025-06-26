import gradio as gr
import pandas as pd
import json
import re
from langchain_community.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from transformers import pipeline, AutoTokenizer
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline

# Initialize embedding model (free)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def load_course_db():
    try:
        df = pd.read_csv("courses.csv")
        print(f"Loaded CSV with columns: {df.columns.tolist()}")
        texts = []
        for _, row in df.iterrows():
            title = row.get('title') or row.get('course_title') or "Untitled Course"
            description = row.get('description') or row.get('course_description') or "No description"
            url = row.get('url') or row.get('course_url') or "#"
            text = f"{title}: {description} | URL: {url}"
            texts.append(text)
        return Chroma.from_texts(texts, embeddings), df
    except Exception as e:
        print(f"Error loading CSV: {str(e)}")
        sample_courses = [
            {"title": "Python Fundamentals", "description": "Learn programming", "url": "#"},
            {"title": "Machine Learning", "description": "ML techniques", "url": "#"},
        ]
        df = pd.DataFrame(sample_courses)
        texts = [f"{row['title']}: {row['description']} | URL: {row['url']}" for _, row in df.iterrows()]
        return Chroma.from_texts(texts, embeddings), df

# Load course data
vector_db, course_df = load_course_db()

# Initialize FREE local model (DistilGPT2)
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
generator = pipeline(
    "text-generation",
    model="distilgpt2",  # Free, runs locally
    max_new_tokens=200,
    max_length=1024,
    truncation=True,
    temperature=0.5
)
llm = HuggingFacePipeline(pipeline=generator)

def recommend_courses(query):
    try:
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_db.as_retriever(search_kwargs={"k": 3})
        )
        
        prompt = f"""
        Based on user background: "{query}"
        Recommend 3 courses with:
        - Title
        - Reason: Brief justification
        - Direct URL
        Format as JSON list: [{{"title": "...", "reason": "...", "url": "..."}}]
        """
        
        result = qa.invoke({"query": prompt})
        response = result["result"]
        
        # Enhanced JSON parsing
        try:
            json_match = re.search(r'\[\s*\{.*?\}\s*(?:,\s*\{.*?\}\s*)*\]', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            # Fallback: Manual extraction
            courses = []
            pattern = r'"title":\s*"([^"]+)".*?"reason":\s*"([^"]+)".*?"url":\s*"([^"]+)"'
            for match in re.finditer(pattern, response, re.DOTALL):
                courses.append({
                    "title": match.group(1),
                    "reason": match.group(2),
                    "url": match.group(3)
                })
            return courses if courses else [{"error": "Could not parse courses"}]
        except:
            return [{"error": "JSON parsing failed"}]
            
    except Exception as e:
        return [{"error": f"Recommendation failed: {str(e)}"}]

def generate_learning_path(recommendations):
    try:
        if not recommendations or not recommendations.strip():
            return {"error": "Please provide course names"}
            
        prompt = f"""
        Create a 3-month learning plan for these courses: {recommendations}
        Include:
        - Weekly milestones
        - Project suggestions
        - Skills to develop
        Format response as a structured JSON object
        """
        
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_db.as_retriever()
        )
        
        result = qa.invoke({"query": prompt})
        return {"learning_path": result["result"]}
        
    except Exception as e:
        return {"error": f"Learning path generation failed: {str(e)}"}

# Gradio interface
with gr.Blocks(theme=gr.themes.Soft(), title="Free Course Bot") as demo:
    gr.Markdown("# 🎓 Free Course Recommender")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Get Course Recommendations")
            background = gr.Textbox(
                label="Your background/goals", 
                placeholder="e.g., 'CS student interested in AI'",
                lines=2
            )
            rec_btn = gr.Button("Get Recommendations", variant="primary")
            rec_output = gr.JSON(label="Recommended Courses")
        
        with gr.Column():
            gr.Markdown("### Create Learning Path")
            gr.Markdown("Enter course names from recommendations")
            path_input = gr.Textbox(
                label="Courses (comma separated)",
                placeholder="e.g., Python, Machine Learning"
            )
            path_btn = gr.Button("Generate Learning Path", variant="primary")
            path_output = gr.JSON(label="Personalized Plan")
    
    rec_btn.click(
        fn=recommend_courses,
        inputs=background,
        outputs=rec_output
    )
    
    path_btn.click(
        fn=generate_learning_path,
        inputs=path_input,
        outputs=path_output
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
