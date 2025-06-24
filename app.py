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
        # Load massive MOOC dataset (23,000+ courses)
        dataset = load_dataset("mitul1999/online-courses-usage-and-history-dataset")
        df = dataset['train'].to_pandas()
        
        # Create combined text for embeddings
        texts = [
            f"{row['course_title']}: {row['course_description']} | Category: {row['category']} | Level: {row['level']} | Platform: {row['platform']}"
            for _, row in df.iterrows()
        ]
        return Chroma.from_texts(texts, embeddings, collection_name="courses"), df
    except Exception as e:
        print(f"Error loading main dataset: {str(e)}")
        try:
            # Fallback to MOOC dataset
            url = "https://raw.githubusercontent.com/Bladefidz/data-mining/master/coursera/text-retrieval-and-search-engines/notes/mooc.dat"
            response = requests.get(url)
            response.raise_for_status()
            
            # Parse custom format
            courses = []
            for line in response.text.split('\n'):
                if ' - ' in line and 'http' in line:
                    parts = line.split(' - ')
                    title = parts[0]
                    rest = parts[1].split(' ')
                    url = rest[-1]
                    description = ' '.join(rest[:-1])
                    courses.append({
                        'title': title,
                        'description': description,
                        'url': url,
                        'level': 'Intermediate'
                    })
            df = pd.DataFrame(courses)
            texts = [f"{row.title}: {row.description}" for _, row in df.iterrows()]
            return Chroma.from_texts(texts, embeddings, collection_name="courses"), df
        except Exception as e2:
            print(f"Fallback failed: {str(e2)}")
            # Ultimate fallback to sample data
            sample_courses = [
                {"title": "Python Fundamentals", "description": "Learn core programming", 
                 "level": "Beginner", "url": "https://example.com/python"},
                {"title": "Machine Learning", "description": "Deep learning techniques", 
                 "level": "Intermediate", "url": "https://example.com/ml"}
            ]
            df = pd.DataFrame(sample_courses)
            texts = [f"{row.title}: {row.description} | Level: {row.level}" for _, row in df.iterrows()]
            return Chroma.from_texts(texts, embeddings, collection_name="courses"), df

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
            retriever=vector_db.as_retriever(search_kwargs={"k": 5})
        )
        
        prompt = f"""
        Based on user background: "{query}"
        Recommend 3 personalized courses from our database of 23,000+ options.
        For each course, provide:
        - Title
        - Reason: Brief justification matching user's background
        - Difficulty level
        - Direct URL
        Format as JSON list
        Example: [{{"title": "...", "reason": "...", "level": "...", "url": "..."}}]
        """
        
        response = qa.run(prompt)
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
        Create a comprehensive 3-month learning path with:
        - Weekly milestones
        - Hands-on project suggestions
        - Skill validation metrics
        - Estimated time commitment
        Format as JSON with keys: weeks, projects, metrics, time_commitment
        """
        
        return qa.run(prompt)
    except Exception as e:
        return {"error": str(e)}

with gr.Blocks(theme=gr.themes.Soft(), title="Course Recommendation Bot") as demo:
    gr.Markdown("# 🎓 Comprehensive Course Advisor")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Get Personalized Recommendations")
            gr.Markdown("Our database contains 23,000+ courses from Coursera, edX, Udemy, and more")
            background = gr.Textbox(label="Describe your background/goals", 
                                   placeholder="e.g., 'High school graduate interested in AI career'")
            rec_btn = gr.Button("Find My Courses", variant="primary")
            rec_output = gr.JSON(label="Recommended Courses")
        
        with gr.Column():
            gr.Markdown("### Generate Learning Path")
            gr.Markdown("Create a customized study plan")
            path_input = gr.Textbox(label="Based on these courses")
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
