import gradio as gr
import pandas as pd
import json
import re
from langchain_community.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from transformers import pipeline, AutoTokenizer
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline

# Initialize embedding model (free)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def load_course_db():
    try:
        # List all your CSV files
        csv_files = [
            ("courses.csv", "Generic"),  # Your original dataset
            ("coursera_data.csv", "Coursera"),
            ("udemy_courses.csv", "Udemy")
        ]
        all_courses = []
        for filename, platform in csv_files:
            try:
                df = pd.read_csv(filename)
                print(f"Loaded {filename} with columns: {df.columns.tolist()}")
                for _, row in df.iterrows():
                    title = row.get('title') or row.get('course_title') or "Untitled Course"
                    description = row.get('description') or row.get('course_description') or row.get('subject') or "No description"
                    url = row.get('url') or row.get('course_url') or row.get('link') or "#"
                    # Use the platform from the file or from the row if present
                    plat = row.get('platform') or platform
                    text = f"{title}: {description} | URL: {url} | Platform: {plat}"
                    all_courses.append({
                        "title": title,
                        "description": description,
                        "url": url,
                        "platform": plat,
                        "text": text
                    })
            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")
        if not all_courses:
            raise ValueError("No courses loaded from any file.")
        # Build vector DB
        texts = [c["text"] for c in all_courses]
        df = pd.DataFrame(all_courses)
        return Chroma.from_texts(texts, embeddings), df
    except Exception as e:
        print(f"Error loading datasets: {str(e)}")
        # Fallback to sample data
        sample_courses = [
            {"title": "Python Fundamentals", "description": "Learn programming", "url": "#", "platform": "Sample"},
            {"title": "Machine Learning", "description": "ML techniques", "url": "#", "platform": "Sample"},
        ]
        df = pd.DataFrame(sample_courses)
        texts = [f"{row['title']}: {row['description']} | URL: {row['url']} | Platform: {row['platform']}" for _, row in df.iterrows()]
        return Chroma.from_texts(texts, embeddings), df


# Load course data
vector_db, course_df = load_course_db()

# Initialize FREE local model (DistilGPT2)
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
generator = pipeline(
    "text-generation",
    model="distilgpt2",  # Free, runs locally
    max_new_tokens=200,
    max_length=1024,
    truncation=True,
    temperature=0.5
)
llm = HuggingFacePipeline(pipeline=generator)

def recommend_courses(query):
    try:
        # Retrieve top 3 relevant courses
        retrieved = vector_db.similarity_search(query, k=3)
        courses = []
        for doc in retrieved:
            # Parse the course info from doc.page_content
            match = re.match(r'^(.*?): (.*?) \| URL: (.*)$', doc.page_content)
            if match:
                title = match.group(1)
                description = match.group(2)
                url = match.group(3)
                courses.append({
                    "title": title,
                    "reason": description,
                    "url": url
                })
        return courses if courses else [{"error": "No courses found"}]
    except Exception as e:
        return [{"error": f"System error: {str(e)}"}]


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
with gr.Blocks(theme=gr.themes.Soft(), title="Free Course Bot") as demo:
    gr.Markdown("# 🎓 Free Course Recommender")
    
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
    demo.launch(server_name="0.0.0.0", server_port=7860)
