import gradio as gr
import pandas as pd
import re
import os
import time
import json
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
LEARNING_PATH_CSV = "Learning_Pathway_Index.csv"  # Fixed filename with capital 'P'

def load_learning_paths():
    """Load and structure learning paths from CSV"""
    try:
        # Check if file exists
        if not os.path.exists(LEARNING_PATH_CSV):
            print(f"Learning path file not found: {LEARNING_PATH_CSV}")
            return {}
        
        df = pd.read_csv(LEARNING_PATH_CSV)
        # Group by Module_Code to create complete paths
        grouped = df.groupby('Module_Code')
        paths = {}
        for name, group in grouped:
            path_name = group['Course_Learning_Material'].iloc[0]
            paths[path_name] = []
            for _, row in group.iterrows():
                paths[path_name].append({
                    "title": row['Module'],
                    "description": row['Course_Learning_Material'],
                    "duration": row['Duration'],
                    "difficulty": row['Difficulty_Level'],
                    "url": row['Links'],
                    "keywords": row['Keywords_Tags_Skills_Interests_Categories']
                })
        return paths
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
            # Skip files that don't exist
            if not os.path.exists(filename):
                print(f"Skipping missing file: {filename}")
                continue
                
            df = pd.read_csv(filename)
            for _, row in df.iterrows():
                title = row.get('title') or row.get('course_title') or "Untitled Course"
                description = row.get('description') or row.get('course_description') or row.get('subject') or "No description"
                url = row.get('url') or row.get('link') or "#"
                
                # Generate URL if missing
                if url == "#" or pd.isna(url):
                    slug = re.sub(r'[^\w\s-]', '', title).strip().lower().replace(' ', '-')
                    if platform == "Coursera":
                        url = f"https://www.coursera.org/learn/{slug}"
                    elif platform == "Udemy":
                        url = f"https://www.udemy.com/course/{slug}/"
                    elif platform == "edX":
                        url = f"https://www.edx.org/course/{slug}"
                
                # Improve description if missing
                if description in ["No description", "", None]:
                    description = f"Comprehensive course on {title.split(':')[0]}"
                
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
    
    if not all_courses:
        # Use learning paths as fallback if no courses found
        return []
    
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
                {"title": "Machine Learning", "description": "ML techniques", "url": "#", "platform": "Sample", "text": "Machine Learning: ML techniques | URL: # | Platform: Sample"},
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
        
        # Save combined courses to CSV
        df = pd.DataFrame(all_courses)
        df.to_csv(COMBINED_CSV, index=False)
        
        print(f"Vector database built in {time.time()-start_time:.2f} seconds")
        return vector_db, df
    else:
        print("Loading precomputed vector database...")
        vector_db = Chroma(
            persist_directory=VECTOR_DB_DIR,
            embedding_function=embeddings
        )
        if os.path.exists(COMBINED_CSV):
            course_df = pd.read_csv(COMBINED_CSV)
        else:
            course_df = pd.DataFrame()
        return vector_db, course_df

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
                
                # Ensure URL is valid
                if url in ["#", "", None]:
                    slug = re.sub(r'[^\w\s-]', '', title).strip().lower().replace(' ', '-')
                    if platform == "Coursera":
                        url = f"https://www.coursera.org/learn/{slug}"
                    elif platform == "Udemy":
                        url = f"https://www.udemy.com/course/{slug}/"
                    elif platform == "edX":
                        url = f"https://www.edx.org/course/{slug}"
                
                # Improve reason if missing
                if reason in ["No description", "", None]:
                    reason = f"Comprehensive course on {title.split(':')[0]}"
                
                courses.append({
                    "title": title,
                    "reason": reason,
                    "url": url,
                    "platform": platform
                })
        
        return courses if courses else [{"error": "No courses found"}]
    
    except Exception as e:
        return [{"error": f"System error: {str(e)}"}]

def generate_learning_path(recommendations):
    """Generate learning path for selected courses with robust validation"""
    try:
        if not recommendations or not recommendations.strip():
            return {"error": "Please provide course names"}
        
        # Limit to 3 courses to avoid overwhelming the model
        courses = [c.strip() for c in recommendations.split(",")][:3]
        course_list = ", ".join(courses)
        
        # Enhanced prompt with JSON structure example
        prompt = f"""
        Create a detailed 3-month learning plan for these courses: {course_list}
        Structure your response as a JSON object with these keys:
        - "overview": (1-paragraph summary of the learning journey)
        - "weekly_schedule": (list of 12 weekly plans with topics/resources)
        - "projects": (list of 3 milestone projects)
        - "skills_developed": (list of skills gained)
        - "resources": (list of relevant resource URLs)
        
        Example valid JSON format:
        {{
          "overview": "This path teaches...",
          "weekly_schedule": [
            "Week 1: Topic A - Resource1, Resource2",
            "Week 2: Topic B - Resource3"
          ],
          "projects": [
            "Project 1: Description",
            "Project 2: Description"
          ],
          "skills_developed": ["Skill1", "Skill2"],
          "resources": ["https://resource1", "https://resource2"]
        }}
        
        Important:
        - Focus only on the provided courses
        - Include practical projects
        - Suggest real learning resources
        - Maintain logical progression
        """
        
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_db.as_retriever()
        )
        
        result = qa.invoke({"query": prompt})
        response = result["result"]
        
        # Attempt to extract JSON
        try:
            # Look for JSON code block
            json_match = re.search(r'``````', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            
            # Try parsing as raw JSON
            return json.loads(response)
        except:
            # Fallback: Use predefined path if available
            for path_name in learning_paths.keys():
                if any(course.lower() in path_name.lower() for course in courses):
                    return {
                        "path_name": path_name,
                        "steps": learning_paths[path_name],
                        "note": "Custom path unavailable. Showing closest match."
                    }
            
            # Final fallback: Error message
            return {
                "error": "Couldn't generate custom path",
                "suggestion": "Try fewer courses or select a predefined path",
                "raw_response": response[:500] + "..." if len(response) > 500 else response
            }
        
    except Exception as e:
        return {"error": f"Learning path generation failed: {str(e)}"}


# Gradio interface
with gr.Blocks(theme=gr.themes.Soft(), title="Course Learning Advisor") as demo:
    gr.Markdown("# 🎓 Course Learning Advisor")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Get Course Recommendations")
            background = gr.Textbox(
                label="Your learning goals", 
                placeholder="e.g., 'Python for beginners' or 'Advanced machine learning'",
                lines=2
            )
            rec_btn = gr.Button("Get Recommendations", variant="primary")
            rec_output = gr.JSON(label="Recommended Courses")
        
        with gr.Column():
            gr.Markdown("### Create Learning Path")
            gr.Markdown("**Predefined Paths**: Select from available learning paths")
            
            # Dropdown for learning paths
            path_options = list(learning_paths.keys())
            path_dropdown = gr.Dropdown(
                choices=path_options,
                label="Select a learning path",
                interactive=True
            )
            
            gr.Markdown("**Custom Path**: Enter course names from recommendations")
            path_input = gr.Textbox(
                label="Courses (comma separated)",
                placeholder="e.g., Python, Machine Learning"
            )
            path_btn = gr.Button("Generate Learning Path", variant="primary")
            path_output = gr.JSON(label="Learning Plan")
    
    # Event handlers
    rec_btn.click(
        fn=recommend_courses,
        inputs=background,
        outputs=rec_output
    )
    
    path_dropdown.change(
        fn=lambda x: {"path_name": x, "steps": learning_paths.get(x, [])},
        inputs=path_dropdown,
        outputs=path_output
    )
    
    path_btn.click(
        fn=generate_learning_path,
        inputs=path_input,
        outputs=path_output
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860) 
