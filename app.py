import gradio as gr
import pandas as pd
import os
import json
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from huggingface_hub import login

# Initialize HF token
login(token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))

# Initialize embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def load_course_db():
    try:
        # Try loading from local CSV first
        df = pd.read_csv("courses.csv")
        print(f"Loaded CSV with columns: {df.columns.tolist()}")
        
        # Check what columns actually exist and adapt
        if 'title' in df.columns:
            title_col = 'title'
        elif 'course_title' in df.columns:
            title_col = 'course_title'
        elif 'name' in df.columns:
            title_col = 'name'
        else:
            title_col = df.columns[0]  # Use first column as title
            
        if 'description' in df.columns:
            desc_col = 'description'
        elif 'course_description' in df.columns:
            desc_col = 'course_description'
        elif 'summary' in df.columns:
            desc_col = 'summary'
        else:
            desc_col = df.columns[1] if len(df.columns) > 1 else title_col
        
        # Create texts for embedding
        texts = []
        for _, row in df.iterrows():
            text = f"{row[title_col]}: {row[desc_col]}"
            if 'level' in df.columns:
                text += f" | Level: {row['level']}"
            if 'subject' in df.columns:
                text += f" | Subject: {row['subject']}"
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

# Initialize LLM
llm = HuggingFaceHub(
    repo_id="microsoft/DialoGPT-medium",  # Use a more reliable model
    model_kwargs={"temperature": 0.7, "max_length": 512},
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
        Based on user query: "{query}"
        Recommend 3 relevant courses with:
        - Title
        - Brief reason
        - Difficulty level
        - URL
        Format as JSON: [{{"title": "...", "reason": "...", "level": "...", "url": "..."}}]
        """
        
        # Use invoke instead of run
        result = qa.invoke({"query": prompt})
        response = result["result"]
        
        # Try to parse as JSON, fallback to text
        try:
            return json.loads(response)
        except:
            # Create structured response from text
            return {
                "recommendations": response,
                "note": "Generated response (parsing as JSON failed)"
            }
            
    except Exception as e:
        return {"error": f"Recommendation failed: {str(e)}"}

def generate_learning_path(recommendations):
    try:
        if not recommendations or recommendations.strip() == "":
            return {"error": "Please provide course names"}
            
        prompt = f"""
        Create a 3-month learning plan for: {recommendations}
        Include:
        - Weekly goals
        - Projects
        - Skills to develop
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
    gr.Markdown("# 🎓 Comprehensive Course Advisor")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Get Personalized Recommendations")
            background = gr.Textbox(
                label="Describe your background/goals", 
                placeholder="e.g., 'High school graduate interested in AI'",
                value="high school graduate interested in AI"
            )
            rec_btn = gr.Button("Find My Courses", variant="primary")
            rec_output = gr.JSON(label="Recommended Courses")
        
        with gr.Column():
            gr.Markdown("### Generate Learning Path")
            path_input = gr.Textbox(
                label="Based on these courses (comma separated)",
                placeholder="e.g., 'Python, Machine Learning, Data Science'"
            )
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
