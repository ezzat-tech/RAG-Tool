from transformers import pipeline
from datasets import load_dataset

qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

dataset = load_dataset("squad", split="validation", trust_remote_code=True)
example = dataset[0]
question = example["question"]
context = example["context"]

result = qa_pipeline(question=question, context=context)

print(f"Answer:  {result['answer']}")