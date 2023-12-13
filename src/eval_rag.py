from datasets import load_dataset
import pprint
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AutoConfig, AutoTokenizer, LlamaTokenizer
import torch
import numpy as np
from tqdm.rich import tqdm
from evaluate import load
from sentence_transformers import SentenceTransformer, util

from typing import Optional
from . import gist_llama, gist_t5, weight_diff
from .gist_llama import GistLlamaForCausalLM
from .gist_t5 import GistT5ForConditionalGeneration
import fire

import time
from . import logger_class as logging

logger = logging.create_logger()

torch.inference_mode()
def gist_compress(
    model_name_or_path: str,
    instruction: str,
    input: str = "",
    num_gist_tokens: Optional[int] = 1,
    cache_dir: str = ".cache",
    precision: str = "fp32",
    max_new_tokens: int = 512,
    base_llama_path: Optional[str] = None,
):
    """Decode from a model with gist compression.

    Args:
        model_name_or_path: The model to load. MUST BE A GIST MODEL.
        instruction: The instruction to be compressed (required).
        input: The input for the instruction (optional). Will not be compressed
            or cached.
        num_gist_tokens: number of gist tokens to compress to. This should
            match the number of gist tokens the model was trained on.
        cache_dir: Hugging Face cache dir.
        precision: Precision to load the model in. Recommend fp32 or bf16 to
            save space (not fp16).
        max_new_tokens: Maximum number of new tokens to decode.
        base_llama_path: Any LLaMA model loaded from Hugging Face
            (jayelm/llama-7b-{gist,pos_control,neg_control}-1) is a weight
            diff, not the full model. If loading one of the Hugging Face LLaMA
            models, use this argument to specify the path to the raw LLaMA model.
    """

    is_llama = "llama" in model_name_or_path.lower()
    is_t5 = "t5" in model_name_or_path.lower()

    # Load config
    config = AutoConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir)

    # Load model
    # print(f"Loading model {model_name_or_path}")
    # logger.info(f"Loading model {model_name_or_path}")
    if is_t5:
        model_cls = GistT5ForConditionalGeneration
    elif is_llama:
        model_cls = GistLlamaForCausalLM
    else:
        raise ValueError(f"Model type {model_name_or_path} not supported")

    if model_name_or_path in {
        "jayelm/llama-7b-gist-1",
        "jayelm/llama-7b-pos_control-1",
        "jayelm/llama-7b-neg_control-1",
    }:
        # Load with weight diff file
        if base_llama_path is None:
            raise ValueError(
                f"{model_name_or_path} is a weight diff huggingface repo. "
                "You must specify a `base_llama_path` for this to work."
            )
        else:
            print("Weight diff detected. Applying to original model...")
        model, _ = weight_diff.recover(
            path_raw=base_llama_path,
            path_diff=model_name_or_path,
            test_inference=False,
            cache_dir=cache_dir,
        )
    else:
        model = model_cls.from_pretrained(
            model_name_or_path,
            config=config,
            cache_dir=cache_dir,
        )

    dtypes = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float,
    }
    model = model.to(dtypes[precision]).cuda().eval()

    # Load tokenizer. It must already have gist token defined.
    # print("Loading tokenizer")
    if is_llama:
        tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        assert len(tokenizer) == gist_llama.PRETRAINED_VOCAB_SIZE + 1
        assert model.lm_head.weight.shape[0] == gist_llama.PRETRAINED_VOCAB_SIZE + 1
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        assert len(tokenizer) == gist_t5.PRETRAINED_VOCAB_SIZE + 1
        assert model.shared.weight.shape[0] == gist_t5.PRETRAINED_VOCAB_SIZE + 1
    gist_token = tokenizer.additional_special_tokens_ids[-1]

    # Compress instruction
    # print("Compressing instruction")
    gist_str = "<GIST>" * num_gist_tokens
    prepped_instruction = f"Answer the question using the following context: {instruction}\n{gist_str}"
    instruction_input_ids = tokenizer.encode(prepped_instruction)
    if is_t5:
        instruction_input_ids = instruction_input_ids[:-1]  # Remove eos token
    instruction_input_ids_tensor = (
        torch.tensor(instruction_input_ids).unsqueeze(0).cuda()
    )
    gist_kwargs = {
        "input_ids": instruction_input_ids_tensor,
        "attention_mask": torch.ones_like(instruction_input_ids_tensor),
    }
    if is_llama:
        gist_kwargs["attention_mask_gist"] = torch.ones_like(
            instruction_input_ids_tensor
        )[None, None]
    gist_activations = model.get_gist_activations(
        gist_token=gist_token,
        num_gist_tokens=num_gist_tokens,
        **gist_kwargs,
    )

    # Prepare input. Input decoding must be done carefully: tokenizers will
    # tokenize things differently if the input is at the start of the string
    # (vs if it follows a gist token). The simplest thing to do to ensure
    # consistency is add a dummy gist token before the input, then remove it
    # from the input ids later.
    if is_t5:
        input_suffix = ""
    else:
        input_suffix = "\nOutput:"

    if input:
        prepped_input = f"<GIST>\nInput: {input}{input_suffix}"
        full_prompt = (
            f"Answer the question using the following context: {instruction}\n{gist_str}\nInput: {input}{input_suffix}"
        )
    else:
        prepped_input = f"<GIST>{input_suffix}"
        full_prompt = f"Answer the question using the following context: {instruction}\n{gist_str}{input_suffix}"

    input_ids = tokenizer.encode(prepped_input)
    # Trim off the gist token we added at the beginning.
    input_ids = input_ids[input_ids.index(gist_token) + 1 :]
    input_ids_tensor = torch.tensor(input_ids).unsqueeze(0).cuda()
    attention_mask_with_gist = (
        torch.tensor([1] * (len(input_ids) + num_gist_tokens)).unsqueeze(0).cuda()
    )

    # Sanity check that tokenizing the full prompt is the same as tokenizing the
    # prepped instruction and prepped input separately.
    full_prompt_input_ids = tokenizer.encode(full_prompt)
    assert (
        full_prompt_input_ids == instruction_input_ids + input_ids
    ), "Got different results tokenizing the full prompt vs tokenizing context/input separately"

    # print("Decoding from model")
    gen_kwargs = {
        "input_ids": input_ids_tensor,
        "attention_mask": attention_mask_with_gist,
    }
    if is_llama:
        gen_kwargs["attention_mask_gist"] = attention_mask_with_gist[None, None]
        gen_kwargs["past_key_values"] = gist_activations.past_key_values
        gen_kwargs["gist_offset"] = gist_activations.gist_indices
    else:
        gen_kwargs["gist_activations"] = gist_activations
    tic = time.time()
    generated_tokens = model.generate(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        **gen_kwargs,
    )
    toc = time.time()
    total_time = toc - tic
    output = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    if is_llama:
        output = output[len(prepped_input) - 5 :]
    return output, total_time


def main():
    bertscore = load("bertscore")

    dataset = load_dataset("trivia_qa", "rc.wikipedia")
    train_dataset, val_dataset, test_dataset = dataset['train'], dataset['validation'], dataset['test']
    # pprint.pprint(train_dataset.info.features)
    # print(train_dataset[0]['entity_pages']['wiki_context'][1])

    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base", device_map="auto", torch_dtype=torch.float16)
    # gist_model = "exp/run-gist-1tok-flan-t5-base-wikipedia/run-gist-1tok-flan-t5-base-wikipedia-run-42-lr5e-5/"
    # gist_model = "exp/run-gist-1tok-flan-t5-base-wikipedia/run-gist-1tok-flan-t5-base-wikipedia-run-42/"
    gist_model = "exp/run-gist-2tok-flan-t5-base-alpaca-plus/run-gist-2tok-flan-t5-base-alpaca-plus-run-42/"
    
    num_predictions = 100
    predictions = []
    predictions_rag = []
    predictions_gist = []
    answers = []

    times_vanilla = []
    times_rag = []
    times_gist = []
    val_dataset = val_dataset.shuffle(seed=22)
    dataset_subset = val_dataset.select(range(num_predictions))
    
    for i, row in enumerate(tqdm(dataset_subset, desc="Generating predictions")):
        question = row['question']
        answer = row['answer']['normalized_value']
        answers.append(answer)
        context_str = row['entity_pages']['wiki_context'][0][:1500]

        # print(f'Question: {question}, Answer: {answer}')
        # logger.info(f'Question: {question}, Answer: {answer}')

        # vanilla predictions
        question_context = f"{question}"
        input_ids = tokenizer(question_context, return_tensors="pt").input_ids.to("cuda")
        tic = time.time()
        outputs = model.generate(input_ids, max_new_tokens=10)
        toc = time.time()
        predictions.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
        times_vanilla.append(toc-tic)

        # predictions rag
        question_context = f"Context: {context_str}\n\nQuestion: {question}"
        input_ids = tokenizer(question_context, return_tensors="pt").input_ids.to("cuda")
        tic = time.time()
        outputs = model.generate(input_ids, max_new_tokens=10)
        toc = time.time()
        predictions_rag.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
        times_rag.append(toc-tic)

        #predictions gist
        outputs, total_time = gist_compress(model_name_or_path=gist_model, instruction=question_context, input=question, num_gist_tokens=1)
        predictions_gist.append(outputs)
        times_gist.append(total_time)

    logger.info(f"vanilla: {predictions}")
    logger.info(f"rag: {predictions_rag}")
    logger.info(f"gist: {predictions_gist}")
    logger.info(f"ground truth: {answers}")

    sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    #Compute embedding for both lists
    embeddings_predictions = sentence_model.encode(predictions, convert_to_tensor=True)
    embeddings_predictions_rag = sentence_model.encode(predictions_rag, convert_to_tensor=True)
    embeddings_predictions_gist = sentence_model.encode(predictions_gist, convert_to_tensor=True)
    
    embeddings_answers = sentence_model.encode(answers, convert_to_tensor=True)

    #Compute cosine-similarities
    cosine_scores = torch.nn.functional.cosine_similarity(embeddings_predictions, embeddings_answers)
    cosine_scores_rag = torch.nn.functional.cosine_similarity(embeddings_predictions_rag, embeddings_answers)
    cosine_scores_gist = torch.nn.functional.cosine_similarity(embeddings_predictions_gist, embeddings_answers)
    # logger.info(f"Cosine scores (All): {cosine_scores}")
    # logger.info(f"Cosine scores rag (All): {cosine_scores_rag}")
    # logger.info(f"Cosine scores gist (All): {cosine_scores_gist}")
    logger.info(f"Vanilla mean cosine score: {torch.mean(cosine_scores)}")
    logger.info(f"Rag mean cosine score: {torch.mean(cosine_scores_rag)}")
    logger.info(f"Gist mean cosine score: {torch.mean(cosine_scores_gist)}")
    logger.info("=====================================")
    logger.info(f"Vanilla exact match: {torch.sum(cosine_scores == 1)/len(cosine_scores)}")
    logger.info(f"Rag exact match: {torch.sum(cosine_scores_rag == 1)/len(cosine_scores_rag)}")
    logger.info(f"Gist exact match: {torch.sum(cosine_scores_gist == 1)/len(cosine_scores_gist)}")
    logger.info("=====================================")
    logger.info(f"Times vanilla: {np.mean(times_vanilla)}")
    logger.info(f"Times rag: {np.mean(times_rag)}")
    logger.info(f"Times gist: {np.mean(times_gist)}")


if __name__ == "__main__":
    fire.Fire(main)


"""
Question: Which city does David Soul come from?, Answer: chicago
Question: Who won Super Bowl XX?, Answer: chicago bears
Question: Which was the first European country to abolish capital punishment?, Answer: norway
Question: In which country did he widespread use of ISDN begin in 1988?, Answer: japan
Question: What is Bruce Willis' real first name?, Answer: walter
Question: Which William wrote the novel Lord Of The Flies?, Answer: golding
Question: How is Joan Molinsky better known?, Answer: joan rivers
Question: In which branch of the arts is Patricia Neary famous?, Answer: ballet
['st john s', 'argentina', 'san francisco', 'nfl', 'poland', 'switzerland', 'bruce', 'charles dickens', 'joan molinsky', 'ballet']
['London', 'unanswerable', 'Chicago', 'The Bears', 'Germans', 'unanswerable', 'Walter', 'William Golding', 'Joan Rivers', 'ballerina']
['London', 'Angola', 'New York City', 'Super Bowl XX', 'The United Kingdom', 'United States', 'Bruce Willis', 'William Shakespeare', 'Joan Molinsky is better known for her role as the leader of the Russian Revolutionary Guards in the Soviet Union.', 'The artist is known for her work in the field of painting, drawing, and sculpture.']
['york', 'portugal', 'chicago', 'chicago bears', 'norway', 'japan', 'walter', 'golding', 'joan rivers', 'ballet']
Vanilla mean cosine score:  tensor(0.5549, device='cuda:0')
Rag mean cosine score:  tensor(0.6963, device='cuda:0')
Gist mean cosine score:  tensor(0.4568, device='cuda:0')
"""
# python -m src.compress --model_name_or_path exp/run-gist-2tok-flan-t5-base-alpaca-plus/run-gist-2tok-flan-t5-base-alpaca-plus-run-42/ --instruction "Name the top cities in France that should not be missed. Include the best aspects of each place as well." --input "What is my name?"