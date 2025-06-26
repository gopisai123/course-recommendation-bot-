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
    # ... (keep your existing load_course_db implementation) ...

vector_db, course_df = load_course_db()

# Initialize LLM with JSON-enforcing parameters
llm = HuggingFaceEndpoint(
    endpoint_url="https://api-inference.huggingface.co/models/gpt2",
    task="text-generation",
    temperature=0.3,  # Lower temperature for more predictable output
    max_new_tokens=200,
    repetition_penalty=1.2,  # Reduces repetition
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

def extract_number_from_query(query, default=3):
    # ... (keep your existing implementation) ...

def recommend_courses(query, level):
    try:
        num_courses = extract_number_from_query(query)
        
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
        
        # Revised prompt with JSON schema enforcement
        prompt = f"""
        User background: "{query}"
        Recommend exactly {num_courses} {level}-level courses in JSON format with:
        - title (string)
        - reason (1-sentence string)
        - level (string matching: Beginner/Intermediate/Advanced)
        - url (valid URL string)
        
        Output ONLY a JSON array following this schema:
        [
          {{
            "title": "Course Name",
            "reason": "Brief justification",
            "level": "{level}",
            "url": "https://valid.url"
          }},
          ... (repeat for {num_courses} courses)
        ]
        """
        
        result = qa.invoke({"query": prompt})
        response = result["result"]
        
        # Enhanced JSON extraction
        try:
            # Find first valid JSON array
            json_match = re.search(r'\[\s*\{.*?\}\s*(?:,\s*\{.*?\}\s*)*\]', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            
            # Fallback: Manual key-value extraction
            courses = []
            pattern = r'"title":\s*"([^"]+)".*?"reason":\s*"([^"]+)".*?"level":\s*"([^"]+)".*?"url":\s*"([^"]+)"'
            for match in re.finditer(pattern, response, re.DOTALL):
                courses.append({
                    "title": match.group(1),
                    "reason": match.group(2),
                    "level": match.group(3),
                    "url": match.group(4)
                })
            return courses if courses else [{"error": "No courses found", "raw": response[:100]}]
            
        except Exception as e:
            return [{"error": f"JSON error: {str(e)}", "raw": response[:100]}]
            
    except Exception as e:
        return [{"error": f"System error: {str(e)}"}]

# ... (keep your existing generate_learning_path and Gradio interface) ...

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