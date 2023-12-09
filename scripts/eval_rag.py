from datasets import load_dataset
import pprint
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from tqdm.rich import tqdm
from evaluate import load
from sentence_transformers import SentenceTransformer, util
bertscore = load("bertscore")

dataset = load_dataset("trivia_qa", "rc.wikipedia")
train_dataset, val_dataset, test_dataset = dataset['train'], dataset['validation'], dataset['test']
# pprint.pprint(train_dataset.info.features)
# print(train_dataset[0]['entity_pages']['wiki_context'][1])

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large", device_map="auto", torch_dtype=torch.float16)

num_predictions = 10
predictions = []
predictions_rag = []
answers = []
dataset_subset = train_dataset.select(range(num_predictions))
for i, row in enumerate(tqdm(dataset_subset, desc="Generating predictions")):
    question = row['question']
    answers.append(row['answer']['normalized_value'])

    # vanilla predictions
    input_ids = tokenizer(question, return_tensors="pt").input_ids.to("cuda")
    outputs = model.generate(input_ids, max_new_tokens=100)
    predictions.append(tokenizer.decode(outputs[0], skip_special_tokens=True))

    # predictions rag
    question = f"Context: {row['entity_pages']['wiki_context'][0][:200]}\n\nQuestion: {question}"
    input_ids = tokenizer(question, return_tensors="pt").input_ids.to("cuda")
    outputs = model.generate(input_ids, max_new_tokens=100)
    predictions_rag.append(tokenizer.decode(outputs[0], skip_special_tokens=True))

print(predictions)
print(predictions_rag)
print(answers)

sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

#Compute embedding for both lists
embeddings_predictions = sentence_model.encode(predictions, convert_to_tensor=True)
embeddings_predictions_rag = sentence_model.encode(predictions_rag, convert_to_tensor=True)
embeddings_answers = sentence_model.encode(answers, convert_to_tensor=True)

#Compute cosine-similarits
cosine_scores = torch.nn.functional.cosine_similarity(embeddings_predictions, embeddings_answers)
cosine_scores_rag = torch.nn.functional.cosine_similarity(embeddings_predictions_rag, embeddings_answers)
print(cosine_scores)
print(cosine_scores_rag)

