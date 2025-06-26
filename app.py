import gradio as gr
import pandas as pd
import json
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from transformers import pipeline, AutoTokenizer
from langchain_community.llms import HuggingFacePipeline

# Initialize embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def load_course_db():
    try:
        df = pd.read_csv("courses.csv")
        print(f"Loaded CSV with columns: {df.columns.tolist()}")
        texts = []
        for _, row in df.iterrows():
            title = row.get('title') or row.get('course_title') or row.get('name') or "Untitled Course"
            description = row.get('description') or row.get('course_description') or row.get('summary') or "No description available"
            # Truncate description to avoid long inputs
            if len(str(description)) > 300:
                description = str(description)[:297] + "..."
            level = row.get('Level') or row.get('level') or "Not specified"
            subject = row.get('subject') or "General"
            url = row.get('url') or row.get('course_url') or "#"
            text = f"{title}: {description} | Level: {level} | Subject: {subject} | URL: {url}"
            texts.append(text)
        return Chroma.from_texts(texts, embeddings), df
    except Exception as e:
        print(f"Error loading CSV: {str(e)}")
        # Fallback to sample data
        sample_courses = [
            {"title": "Python Fundamentals", "description": "Learn core programming concepts", "level": "Beginner", "url": "https://example.com/python"},
            {"title": "Machine Learning", "description": "Deep learning and ML techniques", "level": "Intermediate", "url": "https://example.com/ml"},
            {"title": "Data Science", "description": "Data analysis and visualization", "level": "Intermediate", "url": "https://example.com/ds"},
            {"title": "Web Development", "description": "Full-stack web development", "level": "Beginner", "url": "https://example.com/web"},
            {"title": "Cybersecurity", "description": "Network security and ethical hacking", "level": "Advanced", "url": "https://example.com/cyber"}
        ]
        df = pd.DataFrame(sample_courses)
        texts = [f"{row['title']}: {row['description']} | Level: {row['level']} | URL: {row['url']}" for _, row in df.iterrows()]
        return Chroma.from_texts(texts, embeddings), df

vector_db, course_df = load_course_db()

# Initialize local LLM (distilgpt2 is small & fast)
generator = pipeline(
    "text-generation",
    model="distilgpt2",
    max_new_tokens=128,
    max_length=1024,
    truncation=True,
    temperature=0.5
)
llm = HuggingFacePipeline(pipeline=generator)
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

def recommend_courses(query):
    try:
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_db.as_retriever(search_kwargs={"k": 3})
        )
        prompt = f"""
        User background: "{query}"
        Recommend 3 courses with:
        - Title
        - Reason (1 sentence)
        - Difficulty
        - URL
        Format: [{{"title":"...", "reason":"...", "level":"...", "url":"..."}}]
        """
        # Token length check
        tokens = tokenizer(prompt, return_tensors="pt").input_ids
        if tokens.shape[1] > 900:  # Leave space for generation
            return {"error": "Your input is too long. Please shorten your background description."}
        result = qa.invoke({"query": prompt})
        response = result["result"]
        # Try to parse JSON, fallback to text
        try:
            return json.loads(response)
        except Exception:
            return {"recommendations": response}
    except Exception as e:
        return {"error": f"Recommendation failed: {str(e)}"}

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
        - Estimated time commitment
        Format as a structured JSON object.
        """
        tokens = tokenizer(prompt, return_tensors="pt").input_ids
        if tokens.shape[1] > 900:
            return {"error": "Input too long. Please shorten the course list."}
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_db.as_retriever()
        )
        result = qa.invoke({"query": prompt})
        try:
            return json.loads(result["result"])
        except Exception:
            return {"learning_path": result["result"]}
    except Exception as e:
        return {"error": f"Learning path generation failed: {str(e)}"}

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
