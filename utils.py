import hashlib
import json
import os
import time

import openai
from openai import OpenAI

HARRY_POTTER_PROMPT_TEMPLATE = """
You are a die-hard fan of the Harry Potter series. Every question you are asked, you respond with a short, magical reference to the series.

Do not cite the book or chapter.
Answer briefly and enchantingly.
One or two sentences is good enough:

{question}
"""

# Set the OpenAI API key
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"
api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = openai.OpenAI(api_key=api_key)


def get_magical_answer(question, model_name):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": HARRY_POTTER_PROMPT_TEMPLATE.format(question=question),
            }
        ],
        model=model_name,
    )
    return chat_completion.choices[0].message.content


def openai_get_answer_job(question, output_dir, openai_key, get_answer_f=get_magical_answer, model_name="gpt-4"):
    answer = get_answer_f(question, model_name)
    filename = hashlib.md5((answer + question + str(time.time())).encode()).hexdigest()
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, f"{filename}.json"), "w") as fp:
        json.dump({"question": question, "answer": answer}, fp)


