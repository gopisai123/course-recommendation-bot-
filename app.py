import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"  # Disable Chroma telemetry

import gradio as gr
import pandas as pd
import re
import difflib
from langchain_community.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from transformers import pipeline, AutoTokenizer
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA

# Initialize embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load learning paths from CSV
LEARNING_PATHS_CSV = "top_100_courses_learning_paths.csv"
learning_paths_dict = {}
learning_path_names = []

def safe_split(value, delimiter=';'):
    if pd.isna(value) or value == "":
        return []
    return [v.strip() for v in str(value).split(delimiter) if v.strip()]

if os.path.exists(LEARNING_PATHS_CSV):
    learning_paths_df = pd.read_csv(LEARNING_PATHS_CSV)
    for _, row in learning_paths_df.iterrows():
        course_name = row['course_name'].strip().lower()
        learning_paths_dict[course_name] = {
            "overview": row.get('overview', ''),
            "timeline": safe_split(row.get('timeline')),
            "projects": safe_split(row.get('projects')),
            "resources": safe_split(row.get('resources'))
        }
        learning_path_names.append(course_name)
    print(f"Loaded {len(learning_paths_dict)} learning paths from CSV")
    # Create vector store for path names
    path_vector_db = Chroma.from_texts(
        texts=learning_path_names,
        embedding=embeddings
    )
else:
    print(f"Warning: Learning paths file {LEARNING_PATHS_CSV} not found")
    path_vector_db = None

def load_course_db():
    try:
        csv_files = [
            ("courses.csv", "General"),
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
                    # Enhanced URL validation and generation
                    if url == "#" or pd.isna(url) or not str(url).startswith("http"):
                        slug = re.sub(r'[^\w\s-]', '', title).strip().lower().replace(' ', '-')
                        if platform == "Coursera":
                            url = f"https://www.coursera.org/learn/{slug}"
                        elif platform == "Udemy":
                            url = f"https://www.udemy.com/course/{slug}/"
                        else:
                            url = f"https://example.com/course/{slug}" if slug else "#"
                    text = f"TITLE: {title} | DESCRIPTION: {description} | URL: {url} | PLATFORM: {platform}"
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
        texts = [c["text"] for c in all_courses]
        df = pd.DataFrame(all_courses)
        return Chroma.from_texts(texts, embeddings), df
    except Exception as e:
        print(f"Error loading datasets: {str(e)}")
        sample_courses = [
            {"title": "Machine Learning", "description": "Introduction to ML algorithms", "url": "https://example.com/ml", "platform": "Sample"},
            {"title": "Data Science Fundamentals", "description": "Core data science concepts", "url": "https://example.com/ds", "platform": "Sample"},
            {"title": "Python Programming", "description": "Learn Python from scratch", "url": "https://example.com/python", "platform": "Sample"},
        ]
        df = pd.DataFrame(sample_courses)
        texts = [f"TITLE: {row['title']} | DESCRIPTION: {row['description']} | URL: {row['url']} | PLATFORM: {row['platform']}" for _, row in df.iterrows()]
        return Chroma.from_texts(texts, embeddings), df

# Load course data
vector_db, course_df = load_course_db()

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
    try:
        retrieved = vector_db.similarity_search(query, k=5)
        courses = []
        for doc in retrieved:
            content = doc.page_content
            # Robust parsing with error handling
            title_match = re.search(r'TITLE: (.*?) \| DESCRIPTION:', content)
            desc_match = re.search(r'DESCRIPTION: (.*?) \| URL:', content)
            url_match = re.search(r'URL: (.*?) \| PLATFORM:', content)
            platform_match = re.search(r'PLATFORM: (.*?)$', content)
            if all([title_match, desc_match, url_match, platform_match]):
                title = title_match.group(1).strip()
                description = desc_match.group(1).strip()
                url = url_match.group(1).strip()
                platform = platform_match.group(1).strip()
                # Only include courses with valid URLs
                if url.startswith("http"):
                    courses.append({
                        "title": title,
                        "reason": description,
                        "url": url,
                        "platform": platform
                    })
                else:
                    print(f"Skipping course with invalid URL: {title} - {url}")
        if courses:
            return courses
        else:
            return [{"error": "No valid courses found. Try different keywords."}]
    except Exception as e:
        return [{"error": f"System error: {str(e)}"}]

def generate_learning_path(recommendations):
    try:
        if not recommendations.strip():
            return {"error": "Please provide course names"}
        course_name = recommendations.split(",")[0].strip().lower()
        # 1. Semantic search in learning paths
        if path_vector_db:
            results = path_vector_db.similarity_search(course_name, k=1)
            if results:
                matched_path = results[0].page_content
                if matched_path in learning_paths_dict:
                    return {
                        "course": course_name,
                        "matched_path": matched_path,
                        "path": learning_paths_dict[matched_path]
                    }
        # 2. Fuzzy match
        matches = difflib.get_close_matches(
            course_name, 
            learning_paths_dict.keys(), 
            n=1, 
            cutoff=0.6
        )
        if matches:
            return {
                "course": course_name,
                "matched_path": matches[0],
                "path": learning_paths_dict[matches[0]]
            }
        # 3. Fallback to LLM
        prompt = f"""
        Create a 3-month learning plan for: {course_name}
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
with gr.Blocks(theme=gr.themes.Soft(), title="Course Learning Advisor") as demo:
    gr.Markdown("# 🎓 Course Learning Advisor")
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Get Course Recommendations")
            background = gr.Textbox(
                label="Your learning goals", 
                placeholder="e.g., 'Machine learning' or 'Data Science'",
                lines=2
            )
            rec_btn = gr.Button("Get Recommendations", variant="primary")
            rec_output = gr.JSON(label="Recommended Courses")
        with gr.Column():
            gr.Markdown("### Get Learning Roadmap")
            path_input = gr.Textbox(
                label="Course name",
                placeholder="e.g., Machine Learning"
            )
            path_btn = gr.Button("Generate Learning Path", variant="primary")
            path_output = gr.JSON(label="Learning Roadmap")
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
