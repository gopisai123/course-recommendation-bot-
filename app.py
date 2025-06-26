import gradio as gr
import pandas as pd
import json
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings  # Updated import
from langchain.chains import RetrievalQA
from transformers import pipeline  # Local model
from langchain_community.llms import HuggingFacePipeline  # Local model

# Initialize embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def load_course_db():
    try:
        # Load course data from CSV
        df = pd.read_csv("courses.csv")
        print(f"Loaded CSV with columns: {df.columns.tolist()}")
        
        # Create texts for embedding
        texts = []
        for _, row in df.iterrows():
            title = row.get('title') or row.get('course_title') or row.get('name') or "Untitled Course"
            description = row.get('description') or row.get('course_description') or row.get('summary') or "No description available"
            level = row.get('Level') or row.get('level') or "Not specified"
            subject = row.get('subject') or "General"
            url = row.get('url') or row.get('course_url') or "#"
            
            text = f"{title}: {description} | Level: {level} | Subject: {subject} | URL: {url}"
            texts.append(text)
            
        return Chroma.from_texts(texts, embeddings), df
        
    except Exception as e:
        print(f"Error loading CSV: {str(e)}")
        # Fallback to sample data
        sample_courses = [
            {"title": "Python Fundamentals", "description": "Learn core programming concepts", 
             "level": "Beginner", "url": "https://example.com/python"},
            {"title": "Machine Learning", "description": "Deep learning and ML techniques", 
             "level": "Intermediate", "url": "https://example.com/ml"},
            {"title": "Data Science", "description": "Data analysis and visualization", 
             "level": "Intermediate", "url": "https://example.com/ds"},
            {"title": "Web Development", "description": "Full-stack web development", 
             "level": "Beginner", "url": "https://example.com/web"},
            {"title": "Cybersecurity", "description": "Network security and ethical hacking", 
             "level": "Advanced", "url": "https://example.com/cyber"}
        ]
        df = pd.DataFrame(sample_courses)
        texts = [f"{row.title}: {row.description} | Level: {row.level} | URL: {row.url}" for _, row in df.iterrows()]
        return Chroma.from_texts(texts, embeddings), df

vector_db, course_df = load_course_db()

# Initialize LOCAL LLM (Free, no API quota)
generator = pipeline(
    "text-generation", 
    model="distilgpt2",  # Smaller & faster than GPT-2
    max_new_tokens=256,
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
        - Difficulty level
        - Direct URL
        Format as JSON list: [{{"title": "...", "reason": "...", "level": "...", "url": "..."}}]
        """
        
        result = qa.invoke({"query": prompt})
        response = result["result"]
        return json.loads(response)
            
    except Exception as e:
        return {"error": f"Recommendation failed: {str(e)}"}

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
        - Estimated time commitment
        Format response as a structured JSON object
        """
        
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_db.as_retriever()
        )
        
        result = qa.invoke({"query": prompt})
        return json.loads(result["result"])
        
    except Exception as e:
        return {"error": f"Learning path generation failed: {str(e)}"}

# Gradio interface
with gr.Blocks(theme=gr.themes.Soft(), title="Course Recommendation Bot") as demo:
    gr.Markdown("# 🎓 Smart Course Advisor")
    
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
