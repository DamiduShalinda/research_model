import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "application/json",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-pro-002",
    generation_config=generation_config,
  system_instruction="When i provide question , ideal answer and student answer ,  check those and provide me a feedback comparing student answer to ideal answer regarding provided question , focused on more how to improve regarding content relavancy and expectancy , don't mention about ideal answer to the user giving this prompt. give the user only feedback ,mostly regarding content and if needed grammar and structure  , structure means it should have a proper start hook , description and conclusion, provide the feedback as concies and usable\nAddress the user as if they are your student. Provide feedback that is kind, encouraging, and constructive.",
)

def gemini_feedback(question, ideal_answer, student_answer):
    input_text = (
        f"Question: {question} "
        f"Ideal Answer: {ideal_answer} "
        f"Student Answer: {student_answer}"
    )
    chat_session = model.start_chat(history=[])
    response = chat_session.send_message(input_text)
    
    try:
        # Parse the JSON response
        response_data = json.loads(response.text)
        # Extract the 'feedback' field if available
        return response_data.get("feedback", "Feedback not found in response")
    except json.JSONDecodeError:
        # If response is not JSON, return the raw text
        return response.text.strip()
