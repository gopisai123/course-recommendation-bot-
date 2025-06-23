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
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
import pandas as pd
import os

# Initialize embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Course database loader
def load_course_db():
    try:
        df = pd.read_csv("courses.csv")
        texts = [f"{row.title}: {row.description} | Level: {row.level}" for _, row in df.iterrows()]
        return Chroma.from_texts(texts, embeddings, collection_name="courses"), df
    except:
        # Fallback to sample data
        sample_courses = [
            {"title": "Python Fundamentals", "description": "Learn core programming", "level": "Beginner", "url": "https://example.com/python"},
            {"title": "Machine Learning", "description": "Deep learning techniques", "level": "Intermediate", "url": "https://example.com/ml"}
        ]
        df = pd.DataFrame(sample_courses)
        texts = [f"{row.title}: {row.description} | Level: {row.level}" for _, row in df.iterrows()]
        return Chroma.from_texts(texts, embeddings, collection_name="courses"), df

# Initialize components
vector_db, course_df = load_course_db()
llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

# Recommendation engine
def recommend_courses(query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(search_kwargs={"k": 3})
    )
    
    prompt = f"""
    Based on user background: {query}
    Recommend courses with:
    - Title and justification
    - Difficulty level
    - Direct URL
    Format as JSON list
    """
    
    return qa.run(prompt)

# Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎓 Course Recommendation Bot")
    
    with gr.Row():
        user_input = gr.Textbox(label="Your background/goals", placeholder="e.g., 'CS student interested in AI'")
        submit_btn = gr.Button("Get Recommendations")
    
    output = gr.JSON(label="Recommended Courses")
    
    submit_btn.click(
        fn=recommend_courses,
        inputs=user_input,
        outputs=output
    )

demo.launch()
