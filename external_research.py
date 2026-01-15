from groq import Groq
import os

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def external_research_answer(prompt):
    completion = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a hospital-grade medical research AI."},
            {"role": "user", "content": prompt}
        ]
    )
    return {"answer": completion.choices[0].message.content}
