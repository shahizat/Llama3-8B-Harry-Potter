import os
import random
from datasets import load_dataset
from redis.client import Redis
from rq import Queue
from utils import openai_get_answer_job, get_magical_answer

NR_OF_QUESTIONS = 25000
MODEL_NAME = "gpt-4o" #model name
OUTPUT_DIR = "./dataset"
os.environ["OPENAI_API_KEY"] ="YOUR_OPENAI_API_KEY"
OPENAI_API_KEY = os.getenv("YOUR_OPENAI_API_KEY")
dataset = load_dataset("simple_questions_v2", split="train")

sample_ids = random.sample(range(0, len(dataset)), NR_OF_QUESTIONS)
sample = dataset.select(sample_ids)

q = Queue(connection=Redis())

for row in sample:
    q.enqueue(
        openai_get_answer_job,
        row["question"],
        OUTPUT_DIR,
        OPENAI_API_KEY,
        get_answer_f=get_magical_answer,
        model_name=MODEL_NAME,
    )
