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
from sklearn.cluster import KMeans
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Check if precomputed vector database exists
VECTOR_DB_DIR = "vector_db"
COMBINED_CSV = "combined_courses.csv"

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
            df = pd.read_csv(filename)
            df['platform'] = platform
            for _, row in df.iterrows():
                # Extract title
                title = row.get('title') or row.get('course_title') or "Untitled Course"
                
                # Extract description
                description = row.get('description') or row.get('course_description') or row.get('subject') or "No description"
                
                # Extract URL
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
        raise ValueError("No courses loaded from any file.")
    
    return all_courses

def initialize_system():
    """Initialize vector database and course data"""
    # Create vector database if not exists
    if not os.path.exists(VECTOR_DB_DIR):
        print("Building vector database...")
        start_time = time.time()
        
        # Build course data
        all_courses = build_course_data()
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
        course_df = pd.read_csv(COMBINED_CSV)
        return vector_db, course_df

# Initialize system
vector_db, course_df = initialize_system()

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

def extract_platform(query):
    """Detect platform request in query (e.g., 'from Udemy')"""
    platforms = ["udemy", "coursera", "edx"]
    query_lower = query.lower()
    for platform in platforms:
        if f"from {platform}" in query_lower or f"on {platform}" in query_lower:
            return platform.capitalize()
    return None

def recommend_courses(query):
    """Recommend courses based on query with platform filtering"""
    try:
        platform = extract_platform(query)
        search_query = re.sub(r'\b(from|on)\s+\w+', '', query, flags=re.IGNORECASE).strip()
        
        if platform:
            # Filter courses for specific platform
            platform_courses = course_df[course_df['platform'].str.lower() == platform.lower()]
            if platform_courses.empty:
                return [{"error": f"No courses found on {platform}"}]
                
            # Create temporary vector DB
            texts = platform_courses['text'].tolist()
            temp_vector_db = Chroma.from_texts(texts, embeddings)
            retrieved = temp_vector_db.similarity_search(search_query, k=3)
        else:
            # Search all platforms
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

def cluster_courses(num_clusters=5):
    """Cluster courses into learning domains"""
    vectorizer = TfidfVectorizer(max_features=1000)
    text_data = course_df['title'] + " " + course_df['description']
    tfidf_matrix = vectorizer.fit_transform(text_data)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(tfidf_matrix)
    
    # Assign clusters to courses
    course_df['cluster'] = clusters
    return course_df

def generate_learning_path(recommendations):
    """Generate learning path for selected courses"""
    try:
        if not recommendations or not recommendations.strip():
            return {"error": "Please provide course names"}
        
        # Enhanced prompt for structured output
        prompt = f"""
        Create a detailed 3-month learning plan for these courses: {recommendations}
        Structure your response as a JSON object with these keys:
        - "overview": (summary of the learning journey)
        - "weekly_schedule": (list of 12 weekly plans with topics/resources)
        - "projects": (list of 3 milestone projects)
        - "skills_developed": (list of skills gained)
        - "resources": (list of relevant resource URLs)
        
        Include these elements:
        - Progressive skill building from basic to advanced
        - Practical project-based learning
        - Time commitment estimates per week
        - Recommended assessments
        """
        
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_db.as_retriever()
        )
        
        result = qa.invoke({"query": prompt})
        response = result["result"]
        
        # Try to extract JSON from response
        try:
            # Look for JSON code block
            json_match = re.search(r'``````', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            else:
                # Try parsing as pure JSON
                return json.loads(response)
        except:
            # Fallback to returning raw text
            return {"learning_path": response}
        
    except Exception as e:
        return {"error": f"Learning path generation failed: {str(e)}"}

def generate_domain_paths():
    """Generate learning paths for each course domain"""
    clustered_df = cluster_courses()
    domain_paths = {}
    
    for cluster_id in clustered_df['cluster'].unique():
        domain_courses = clustered_df[clustered_df['cluster'] == cluster_id]
        domain_name = f"Domain-{cluster_id}"
        
        # Create path for this domain
        prompt = f"""
        Create a 6-month learning path for these courses: {domain_courses['title'].tolist()[:5]}
        Structure as:
        - Foundation phase (2 months)
        - Core skills phase (3 months)
        - Specialization phase (1 month)
        Include key courses, projects, and outcomes.
        """
        
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_db.as_retriever()
        )
        
        result = qa.invoke({"query": prompt})
        domain_paths[domain_name] = result["result"]
    
    return domain_paths

# Gradio interface
with gr.Blocks(theme=gr.themes.Soft(), title="Course Recommendation Bot") as demo:
    gr.Markdown("# 🎓 Smart Course Advisor")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Get Course Recommendations")
            gr.Markdown("**Tip**: Add 'from Udemy' or 'on Coursera' to filter results")
            background = gr.Textbox(
                label="Your background/goals", 
                placeholder="e.g., 'AI from Udemy' or 'Python on Coursera'",
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
    
    # Domain learning paths section
    with gr.Accordion("Explore Domain Learning Paths", open=False):
        gr.Markdown("### Pre-defined Learning Tracks")
        domain_btn = gr.Button("Generate Domain Paths", variant="secondary")
        domain_output = gr.JSON(label="Domain Learning Paths")
        domain_btn.click(
            fn=generate_domain_paths,
            outputs=domain_output
        )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
