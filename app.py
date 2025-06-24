import gradio as gr
import pandas as pd
import os
import json
from langchain_community.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_huggingface.llms import HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from huggingface_hub import login

# Initialize HF token
login(token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))

# Initialize embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def load_course_db():
    try:
        # Load course data from CSV
        df = pd.read_csv("courses.csv")
        print(f"Loaded CSV with columns: {df.columns.tolist()}")
        
        # Create texts for embedding using available columns
        texts = []
        for _, row in df.iterrows():
            # Title column
            title = row.get('title') or row.get('course_title') or row.get('name') or "Untitled Course"
            
            # Description column
            description = row.get('description') or row.get('course_description') or row.get('summary') or "No description available"
            
            # Additional metadata
            level = row.get('Level') or row.get('level') or "Not specified"
            subject = row.get('subject') or "General"
            
            text = f"{title}: {description} | Level: {level} | Subject: {subject}"
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
        texts = [f"{row.title}: {row.description} | Level: {row.level}" for _, row in df.iterrows()]
        return Chroma.from_texts(texts, embeddings), df

vector_db, course_df = load_course_db()

# Initialize LLM with endpoint URL
# Initialize LLM with endpoint URL - FIXED
# More reliable model that works with most tokens
llm = HuggingFaceEndpoint(
    endpoint_url="https://api-inference.huggingface.co/models/google/flan-t5-large",
    task="text-generation",
    temperature=0.5,
    max_new_tokens=512,
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)


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
        
        # Get response from LLM
        result = qa.invoke({"query": prompt})
        response = result["result"]
        
        # Try to parse JSON, fallback to raw text
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"recommendations": response, "note": "Response format invalid"}
            
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
        return {"learning_path": result["result"]}
        
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
    demo.launch(share=True)
