from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model_utils import load_model, predict
from feedback_utils import gemini_feedback

# Initialize FastAPI app and model
app = FastAPI()
model, tokenizer = load_model()

# Request schema
class EvaluationRequest(BaseModel):
    question: str
    ideal_answer: str
    student_answer: str

# Endpoint for prediction
@app.post("/evaluate")
async def evaluate(request: EvaluationRequest):
    input_text = (
        f"Question: {request.question} "
        f"Ideal Answer: {request.ideal_answer} "
        f"Student Answer: {request.student_answer}"
    )
    
    predictions = predict(input_text, model, tokenizer)
    feedback = gemini_feedback(request.question, request.ideal_answer, request.student_answer)
    return {
        "content_relevancy_score": f"{round(predictions['content_relevancy_score'], 3)} out of 3",
        "grammar_score": f"{round(predictions['grammar_score'], 3)} out of 5",
        "structure_score": f"{round(predictions['structure_score'], 3)} out of 5",
        "feedback": feedback
    }
