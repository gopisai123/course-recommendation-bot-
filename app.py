import gradio as gr
import pandas as pd
import re
import os
import json
import time  # ADDED THIS IMPORT
from langchain_community.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from transformers import pipeline, AutoTokenizer
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA

# Initialize embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Configuration
VECTOR_DB_DIR = "vector_db"
COMBINED_CSV = "combined_courses.csv"
LEARNING_PATH_CSV = "Learning_Pathway_Index.csv"

def load_learning_paths():
    """Load and structure learning paths from CSV"""
    try:
        if not os.path.exists(LEARNING_PATH_CSV):
            return {}
        
        df = pd.read_csv(LEARNING_PATH_CSV)
        path_dict = {}
        
        # Group by Module_Code to create complete paths
        for module_code, group in df.groupby('Module_Code'):
            path_name = group['Course_Learning_Material'].iloc[0]
            steps = []
            for i, (_, row) in enumerate(group.iterrows(), 1):
                steps.append({
                    "step": i,
                    "title": row['Module'],
                    "description": row['Course_Learning_Material'],
                    "duration": row['Duration'],
                    "difficulty": row['Difficulty_Level'],
                    "url": row['Links']
                })
            path_dict[path_name] = steps
            
        return path_dict
    except Exception as e:
        print(f"Error loading learning paths: {str(e)}")
        return {}

def build_course_data():
    """Load and combine all course data from CSV files"""
    csv_files = [
        ("courses.csv", "edX"),
        ("coursera_data.csv", "Coursera"),
        ("udemy_courses.csv", "Udemy")
    ]
    
    all_courses = []
    for filename, platform in csv_files:
        try:
            if not os.path.exists(filename):
                print(f"Skipping missing file: {filename}")
                continue
                
            df = pd.read_csv(filename)
            for _, row in df.iterrows():
                title = row.get('title') or row.get('course_title') or "Untitled Course"
                description = row.get('description') or row.get('course_description') or row.get('subject') or "No description"
                url = row.get('url') or row.get('link') or "#"
                
                # Create course text
                text = f"{title}: {description} | URL: {url} | Platform: {platform}"
                
                all_courses.append({
                    "title": title,
                    "description": description,
                    "url": url,
                    "platform": platform,
                    "text": text
                })
        except Exception as e:
            print(f"Error loading {filename}: {str(e)}")
    
    return all_courses

def initialize_system():
    """Initialize vector database and course data"""
    # Create vector database if not exists
    if not os.path.exists(VECTOR_DB_DIR):
        print("Building vector database...")
        start_time = time.time()
        
        # Build course data
        all_courses = build_course_data()
        
        # Check if we have courses to process
        if not all_courses:
            print("No courses found. Using sample data.")
            sample_courses = [
                {"title": "Python Fundamentals", "description": "Learn programming", "url": "#", "platform": "Sample", "text": "Python Fundamentals: Learn programming | URL: # | Platform: Sample"},
            ]
            all_courses = sample_courses
        
        texts = [c["text"] for c in all_courses]
        
        # Create and persist vector DB
        vector_db = Chroma.from_texts(
            texts=texts,
            embedding=embeddings,
            persist_directory=VECTOR_DB_DIR
        )
        vector_db.persist()
        
        print(f"Vector database built in {time.time()-start_time:.2f} seconds")
        return vector_db, pd.DataFrame(all_courses)
    else:
        print("Loading precomputed vector database...")
        vector_db = Chroma(
            persist_directory=VECTOR_DB_DIR,
            embedding_function=embeddings
        )
        return vector_db, pd.DataFrame()

# Initialize system
vector_db, course_df = initialize_system()

# Load learning paths
learning_paths = load_learning_paths()

# Initialize local model
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
generator = pipeline(
    "text-generation",
    model="distilgpt2",
    max_new_tokens=200,
    max_length=1024,
    truncation=True,
    temperature=0.5
)
llm = HuggingFacePipeline(pipeline=generator)

def recommend_courses(query):
    """Recommend courses based on query"""
    try:
        # Retrieve top 3 relevant courses
        retrieved = vector_db.similarity_search(query, k=3)
        courses = []
        for doc in retrieved:
            # Parse course info
            match = re.match(r'^(.*?): (.*?) \| URL: (.*?) \| Platform: (.*)$', doc.page_content)
            if match:
                title, reason, url, platform = match.groups()
                courses.append({
                    "title": title,
                    "reason": reason,
                    "url": url,
                    "platform": platform
                })
        
        return courses if courses else [{"error": "No courses found"}]
    
    except Exception as e:
        return [{"error": f"System error: {str(e)}"}]

def get_learning_path(course_name):
    """Get learning path for a single course"""
    try:
        # First check if course has a predefined path
        for path_name, steps in learning_paths.items():
            for step in steps:
                if course_name.lower() in step['title'].lower():
                    return {
                        "course": course_name,
                        "path_name": path_name,
                        "steps": steps
                    }
        
        # If no predefined path, generate with LLM
        prompt = f"""
        Create a 3-month learning roadmap for: {course_name}
        Structure as:
        1. Prerequisites (1 month)
        2. Core Learning (1 month)
        3. Advanced Topics (1 month)
        4. Practical Projects
        Include weekly milestones and resource links.
        """
        
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_db.as_retriever()
        )
        
        result = qa.invoke({"query": prompt})
        return {
            "course": course_name,
            "roadmap": result["result"]
        }
        
    except Exception as e:
        return {"error": f"Learning path generation failed: {str(e)}"}

# Gradio interface
with gr.Blocks(theme=gr.themes.Soft(), title="Course Learning Advisor") as demo:
    gr.Markdown("# 🎓 Course Learning Roadmap Generator")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Get Course Recommendations")
            background = gr.Textbox(
                label="What do you want to learn?", 
                placeholder="e.g., 'Python for beginners' or 'Advanced machine learning'",
                lines=2
            )
            rec_btn = gr.Button("Get Recommendations", variant="primary")
            rec_output = gr.JSON(label="Recommended Courses")
        
        with gr.Column():
            gr.Markdown("### Get Learning Roadmap")
            gr.Markdown("Enter a course name to get its learning path")
            course_input = gr.Textbox(
                label="Course name",
                placeholder="e.g., Python Fundamentals"
            )
            path_btn = gr.Button("Generate Learning Roadmap", variant="primary")
            path_output = gr.JSON(label="Learning Roadmap")
    
    rec_btn.click(
        fn=recommend_courses,
        inputs=background,
        outputs=rec_output
    )
    
    path_btn.click(
        fn=get_learning_path,
        inputs=course_input,
        outputs=path_output
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
