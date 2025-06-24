# import gradio as gr
# from huggingface_hub import InferenceClient

# """
# For more information on `huggingface_hub` Inference API support, please check the docs: https://huggingface.co/docs/huggingface_hub/v0.22.2/en/guides/inference
# """
# client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")


# def respond(
#     message,
#     history: list[tuple[str, str]],
#     system_message,
#     max_tokens,
#     temperature,
#     top_p,
# ):
#     messages = [{"role": "system", "content": system_message}]

#     for val in history:
#         if val[0]:
#             messages.append({"role": "user", "content": val[0]})
#         if val[1]:
#             messages.append({"role": "assistant", "content": val[1]})

#     messages.append({"role": "user", "content": message})

#     response = ""

#     for message in client.chat_completion(
#         messages,
#         max_tokens=max_tokens,
#         stream=True,
#         temperature=temperature,
#         top_p=top_p,
#     ):
#         token = message.choices[0].delta.content

#         response += token
#         yield response


# """
# For information on how to customize the ChatInterface, peruse the gradio docs: https://www.gradio.app/docs/chatinterface
# """
# demo = gr.ChatInterface(
#     respond,
#     additional_inputs=[
#         gr.Textbox(value="You are a friendly Chatbot.", label="System message"),
#         gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
#         gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
#         gr.Slider(
#             minimum=0.1,
#             maximum=1.0,
#             value=0.95,
#             step=0.05,
#             label="Top-p (nucleus sampling)",
#         ),
#     ],
# )


# if __name__ == "__main__":
#     demo.launch()


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
        df = pd.read_csv("courses.csv")
        # Create combined text for embeddings
        texts = [
            f"{row.title}: {row.description} | Skills: {row.skills} | Level: {row.level}"
            for _, row in df.iterrows()
        ]
        return Chroma.from_texts(texts, embeddings, collection_name="courses"), df
    except Exception as e:
        print(f"Error loading courses: {str(e)}")
        # Fallback to sample data
        sample_courses = [
            {"title": "Python Fundamentals", "description": "Learn core programming", 
             "skills": "Python, Algorithms", "level": "Beginner", "url": "https://example.com/python"},
            {"title": "Machine Learning", "description": "Deep learning techniques", 
             "skills": "ML, Statistics", "level": "Intermediate", "url": "https://example.com/ml"}
        ]
        df = pd.DataFrame(sample_courses)
        texts = [f"{row.title}: {row.description} | Level: {row.level}" for _, row in df.iterrows()]
        return Chroma.from_texts(texts, embeddings, collection_name="courses"), df

vector_db, course_df = load_course_db()

# Initialize LLM with API token
llm = HuggingFaceHub(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    model_kwargs={"temperature":0.5, "max_length":1024},
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
        Based on user background: "{query}"
        Recommend 3 courses with:
        - Title
        - Reason: Brief justification
        - Skills: Comma-separated list
        - Level: Beginner/Intermediate/Advanced
        - URL: Direct link
        Format as JSON list
        Example: [{{"title": "...", "reason": "...", "skills": "...", "level": "...", "url": "..."}}]
        """
        
        response = qa.run(prompt)
        # Attempt to parse JSON
        try:
            return json.loads(response)
        except:
            return response
    except Exception as e:
        return {"error": str(e)}

def generate_learning_path(recommendations):
    try:
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_db.as_retriever()
        )
        
        prompt = f"""
        Based on these courses: {recommendations}
        Create a 3-month learning path with:
        - Weekly milestones
        - Project suggestions
        - Skill validation metrics
        Format as JSON with keys: weeks, projects, metrics
        """
        
        return qa.run(prompt)
    except Exception as e:
        return {"error": str(e)}

with gr.Blocks(theme=gr.themes.Soft(), title="Course Recommendation Bot") as demo:
    gr.Markdown("# 🎓 Smart Course Advisor")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("## Get Course Recommendations")
            background = gr.Textbox(label="Your background/goals", 
                                   placeholder="e.g., 'CS student interested in AI'")
            rec_btn = gr.Button("Get Recommendations", variant="primary")
            rec_output = gr.JSON(label="Recommended Courses")
        
        with gr.Column():
            gr.Markdown("## Generate Learning Path")
            path_input = gr.Textbox(label="Based on these courses (comma separated or JSON)")
            path_btn = gr.Button("Create Learning Path", variant="primary")
            path_output = gr.JSON(label="Learning Plan")
    
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

demo.launch(share=True)
