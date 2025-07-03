import gradio as gr
import pandas as pd
import re
import os
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
    path_vector_db = Chroma.from_texts(
        texts=learning_path_names,
        embedding=embeddings
    )
else:
    print(f"Warning: Learning paths file {LEARNING_PATHS_CSV} not found")
    path_vector_db = None

def get_best_description(row):
    for key in ['description', 'course_description', 'subject']:
        val = row.get(key)
        if val and isinstance(val, str) and val.strip() and val.strip().lower() != 'no description':
            return val.strip()
    return "No description"

def load_course_db():
    try:
        csv_files = [
            ("courses.csv", "edX")
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
                    description = get_best_description(row)
                    url = row.get('url') or row.get('link') or "#"
                    if url == "#" or pd.isna(url) or not str(url).startswith("http"):
                        slug = re.sub(r'[^\w\s-]', '', title).strip().lower().replace(' ', '-')
                        url = f"https://www.edx.org/course/{slug}" if slug else "#"
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
        texts = [c["text"] for c in all_courses]
        df = pd.DataFrame(all_courses)
        return Chroma.from_texts(texts, embeddings), df
    except Exception as e:
        print(f"Error loading datasets: {str(e)}")
        sample_courses = [
            {"title": "Python Fundamentals", "description": "Learn programming", "url": "#", "platform": "Sample"},
            {"title": "Machine Learning", "description": "ML techniques", "url": "#", "platform": "Sample"},
        ]
        df = pd.DataFrame(sample_courses)
        texts = [f"{row['title']}: {row['description']} | URL: {row['url']} | Platform: {row['platform']}" for _, row in df.iterrows()]
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
        retrieved = vector_db.similarity_search(query, k=3)
        if not retrieved:
            return "No courses found."
        lines = []
        for doc in retrieved:
            match = re.match(r'^(.*?): (.*?) \| URL: (.*?) \| Platform: (.*)$', doc.page_content)
            if match:
                title = match.group(1)
                reason = match.group(2)
                url = match.group(3)
                # Show the actual link (clickable in Markdown)
                lines.append(
                    f"### {title}\n"
                    f"{reason}\n"
                    f"[{url}]({url})\n"
                )
        return "\n---\n".join(lines)
    except Exception as e:
        return f"System error: {str(e)}"

def format_learning_path(path_dict):
    """Format the learning path dictionary as Markdown."""
    if not path_dict:
        return "No learning path found."
    lines = []
    if path_dict.get("overview"):
        lines.append(f"**Overview:** {path_dict['overview']}\n")
    if path_dict.get("timeline"):
        lines.append("**Timeline:**")
        for week in path_dict["timeline"]:
            lines.append(f"- {week}")
    if path_dict.get("projects"):
        lines.append("**Projects:**")
        for proj in path_dict["projects"]:
            lines.append(f"- {proj}")
    if path_dict.get("resources"):
        lines.append("\n**Resources:**")
        for res in path_dict["resources"]:
            lines.append(f"- [{res}]({res})" if res.startswith("http") else f"- {res}")
    return "\n".join(lines)

def generate_learning_path(recommendations):
    try:
        if not recommendations.strip():
            return "Please provide course names"
        course_name = recommendations.split(",")[0].strip().lower()
        # 1. Semantic search in learning paths
        if path_vector_db:
            results = path_vector_db.similarity_search(course_name, k=1)
            if results:
                matched_path = results[0].page_content
                if matched_path in learning_paths_dict:
                    return format_learning_path(learning_paths_dict[matched_path])
        # 2. Fuzzy match
        matches = difflib.get_close_matches(
            course_name, 
            learning_paths_dict.keys(), 
            n=1, 
            cutoff=0.6
        )
        if matches:
            return format_learning_path(learning_paths_dict[matches[0]])
        # 3. Fallback to LLM
        prompt = f"""
        Create a 3-month learning plan for: {course_name}
        Include:
        - Weekly milestones
        - Project suggestions
        - Skills to develop
        Format response as a structured Markdown list
        """
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_db.as_retriever()
        )
        result = qa.invoke({"query": prompt})
        return result["result"]
    except Exception as e:
        return f"Learning path generation failed: {str(e)}"

# Gradio interface
with gr.Blocks(theme=gr.themes.Soft(), title="Course Learning Advisor") as demo:
    gr.Markdown(
        "<h1 style='text-align: center;'>🎓 Course Learning Advisor</h1>"
    )
    with gr.Row():
        with gr.Column():
            gr.Markdown("#### Course Recommendations")
            background = gr.Textbox(
                label="What do you want to learn?",
                placeholder="e.g., Data Science, Machine Learning",
                lines=2
            )
            rec_btn = gr.Button("Get Recommendations", variant="primary")
            rec_output = gr.Markdown(label="Recommended Courses")
        with gr.Column():
            gr.Markdown("#### Personalized Learning Path")
            path_input = gr.Textbox(
                label="Enter a course or topic",
                placeholder="e.g., Python Fundamentals"
            )
            path_btn = gr.Button("Generate Learning Path", variant="primary")
            path_output = gr.Markdown(label="Learning Roadmap")
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
