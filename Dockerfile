# Use the official Python base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code and data
COPY app.py .
COPY courses.csv .

# Expose the port your app will run on (Gradio default is 7860)
EXPOSE 7860

# Start your application
CMD ["python", "app.py"]
