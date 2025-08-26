# Course Learning Advisor 🎓

A **personalized course recommendation and learning path generator** built using **Gradio**, **LangChain**, and **Hugging Face embeddings**.  
It helps users discover the best courses and generate structured learning plans with milestones, projects, and resources.

---

## Deployed Version

Try it online: [Course Recommendation Bot](https://huggingface.co/spaces/GopiSai45/course-recommendation-bot)

---

## Project Overview

- Loads course data from CSV files (`courses.csv` and `top_100_courses_learning_paths.csv`).  
- Uses **semantic search** with Hugging Face embeddings to recommend courses.  
- Generates **personalized learning paths** using:
  - Structured CSV data  
  - Fuzzy string matching  
  - LLM fallback (DistilGPT2 via LangChain)  
- Provides results in an **interactive Gradio web interface**.  

---

## Features

- **Course Recommendations:** Get top courses based on a topic or query.  
- **Learning Paths:** Generate a 3-month roadmap with skills, projects, and resources.  
- **Interactive UI:** Gradio-based interface for easy use.  
- **Multiple Fallbacks:** Ensures recommendations even if exact matches aren’t found.  

---

## How to Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/REPO_NAME.git
cd REPO_NAME
pip install -r requirements.txt
python main.py
