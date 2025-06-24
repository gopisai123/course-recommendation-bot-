import gradio as gr
import pandas as pd
import os
import json
import requests
from io import StringIO
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from huggingface_hub import login
from datasets import load_dataset

# Initialize HF token
login(token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))

# Initialize embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def load_course_db():
    try:
        # Try loading from local CSV first
        df = pd.read_csv("courses.csv")
        texts = [
            f"{row['title']}: {row['description']} | Level: {row['level']} | Skills: {row['skills']}"
            for _, row in df.iterrows()
        ]
        return Chroma.from_texts(texts, embeddings), df
    except Exception as e:
        print(f"Error loading local CSV: {str(e)}")
        try:
            # Load from Hugging Face dataset
            dataset = load_dataset("mitul1999/online-courses-usage-and-history-dataset")
            df = dataset['train'].to_pandas()
            texts = [
                f"{row['name']}: {row['description']} | Category: {row['category']} | Level: {row['level']}"
                for _, row in df.iterrows()
            ]
            return Chroma.from_texts(texts, embeddings), df
        except Exception as e2:
            print(f"Error loading dataset: {str(e2)}")
            # Ultimate fallback to sample data
            sample_courses = [
                {"title": "Python Fundamentals", "description": "Learn core programming", 
                 "level": "Beginner", "url": "https://example.com/python"},
                {"title": "Machine Learning", "description": "Deep learning techniques", 
                 "level": "Intermediate", "url": "https://example.com/ml"}
            ]
            df = pd.DataFrame(sample_courses)
            texts = [f"{row.title}: {row.description} | Level: {row.level}" for _, row in df.iterrows()]
            return Chroma.from_texts(texts, embeddings), df

vector_db, course_df = load_course_db()

# Initialize LLM with API token
llm = HuggingFaceHub(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    model_kwargs={"temperature":0.5, "max_length":1024},
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
        Recommend 3 personalized courses with:
        - Title
        - Reason: Brief justification
        - Difficulty level
        - Direct URL
        Format as JSON list
        Example: [{{"title": "...", "reason": "...", "level": "...", "url": "..."}}]
        """
        
        # Use invoke() instead of run() to avoid deprecation warning
        result = qa.invoke({"query": prompt})
        response = result["result"]
        
        try:
            return json.loads(response)
        except:
            return response
    except Exception as e:
        return {"error": str(e)}

def generate_learning_path(recommendations):
    try:
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_db.as_retriever()
        )
        
        prompt = f"""
        Based on these courses: {recommendations}
        Create a 3-month learning path with:
        - Weekly milestones
        - Project suggestions
        - Skill validation
        Format as JSON with keys: weeks, projects, metrics
        """
        
        # Use invoke() instead of run()
        result = qa.invoke({"query": prompt})
        return result["result"]
    except Exception as e:
        return {"error": str(e)}

with gr.Blocks(theme=gr.themes.Soft(), title="Course Recommendation Bot") as demo:
    gr.Markdown("# 🎓 Comprehensive Course Advisor")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Get Personalized Recommendations")
            background = gr.Textbox(label="Describe your background/goals", 
                                   placeholder="e.g., 'High school graduate interested in AI career'")
            rec_btn = gr.Button("Find My Courses", variant="primary")
            rec_output = gr.JSON(label="Recommended Courses")
        
        with gr.Column():
            gr.Markdown("### Generate Learning Path")
            path_input = gr.Textbox(label="Based on these courses (comma separated)")
            path_btn = gr.Button("Build Learning Path", variant="primary")
            path_output = gr.JSON(label="Personalized Learning Plan")
    
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

demo.launch()
