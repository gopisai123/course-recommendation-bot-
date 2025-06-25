import gradio as gr
import pandas as pd
import os
import json
import re
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
        
        # Create texts and metadata for embedding
        texts = []
        metadatas = []
        for _, row in df.iterrows():
            # Extract data
            title = row.get('title') or row.get('course_title') or row.get('name') or "Untitled Course"
            description = row.get('description') or row.get('course_description') or row.get('summary') or "No description available"
            level = (row.get('Level') or row.get('level') or "Not specified").lower()
            subject = row.get('subject') or "General"
            url = row.get('url') or row.get('course_url') or "#"
            
            # Prepare data for vector store
            text = f"{title}: {description}"
            metadata = {"level": level, "subject": subject, "url": url}
            
            texts.append(text)
            metadatas.append(metadata)
            
        return Chroma.from_texts(texts, embeddings, metadatas=metadatas), df
        
    except Exception as e:
        print(f"Error loading CSV: {str(e)}")
        # Fallback to sample data
        sample_courses = [
            {"title": "Python Fundamentals", "description": "Learn core programming concepts", 
             "level": "beginner", "url": "https://example.com/python"},
            {"title": "Machine Learning", "description": "Deep learning and ML techniques", 
             "level": "intermediate", "url": "https://example.com/ml"},
            {"title": "Data Science", "description": "Data analysis and visualization", 
             "level": "intermediate", "url": "https://example.com/ds"},
            {"title": "Web Development", "description": "Full-stack web development", 
             "level": "beginner", "url": "https://example.com/web"},
            {"title": "Cybersecurity", "description": "Network security and ethical hacking", 
             "level": "advanced", "url": "https://example.com/cyber"}
        ]
        df = pd.DataFrame(sample_courses)
        
        texts = []
        metadatas = []
        for _, row in df.iterrows():
            text = f"{row.title}: {row.description}"
            metadata = {"level": row.level, "url": row.url}
            texts.append(text)
            metadatas.append(metadata)
            
        return Chroma.from_texts(texts, embeddings, metadatas=metadatas), df

vector_db, course_df = load_course_db()

# Initialize LLM with reliable endpoint
llm = HuggingFaceEndpoint(
    endpoint_url="https://api-inference.huggingface.co/models/gpt2",
    task="text-generation",
    temperature=0.7,
    max_new_tokens=128,  # You can increase if needed, but 128 is fast and safe
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

def extract_number_from_query(query, default=3):
    """Extract requested number of courses from user query"""
    match = re.search(r"\b(\d+)\b", query)
    return int(match.group(1)) if match else default

def recommend_courses(query, level):
    try:
        # Determine number of courses requested
        num_courses = extract_number_from_query(query)
        
        # Configure retriever with level filter
        retriever = vector_db.as_retriever(
            search_kwargs={
                "k": num_courses,
                "filter": {"level": level.lower()}
            }
        )
        
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever
        )
        
        prompt = f"""
        Based on user background: "{query}"
        Recommend {num_courses} {level}-level courses with:
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
            # Extract courses from text
            courses = []
            for line in response.split('\n'):
                if line.strip() and ('http' in line or 'https' in line):
                    parts = line.split('-')
                    if len(parts) > 1:
                        courses.append({
                            "title": parts[0].strip(),
                            "reason": parts[1].split('http')[0].strip(),
                            "url": line[line.find('http'):].strip()
                        })
            return courses if courses else {"recommendations": response, "note": "Response format invalid"}
            
    except Exception as e:
        return {"error": f"Recommendation failed: {str(e)}"}

def generate_learning_path(recommendations):
    try:
        if not recommendations or not recommendations.strip():
            return {"error": "Please provide course names"}
            
        prompt = f"""
        Create a practical 3-month learning plan for: {recommendations}
        Include:
        - 12 weekly milestones
        - 2-3 hands-on projects
        - Skills to develop each month
        - Estimated weekly time commitment
        Format response as JSON with keys: ["weeks", "projects", "skills", "time_commitment"]
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
            level_selector = gr.Radio(
                choices=["Beginner", "Intermediate", "Advanced"],
                value="Beginner",
                label="Select difficulty level"
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
        inputs=[background, level_selector],
        outputs=rec_output
    )
    
    path_btn.click(
        fn=generate_learning_path,
        inputs=path_input,
        outputs=path_output
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
