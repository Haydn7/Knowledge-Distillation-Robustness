from transformers import AutoTokenizer
from torch import nn, Tensor
from datasets import Dataset


def tokenize_question(question:str, tokenizer: AutoTokenizer, device: str) -> dict[str, Tensor]:
    if isinstance(question, list):
        messages = [{ "role": "user", "content": q } for q in question]
    else:
        messages = [ {"role": "user", "content": question } ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer([text], return_tensors="pt").to(device)


def question_model(model: nn.Module, tokenizer: AutoTokenizer, question: str, deterministic: bool=False) -> str:
    model_inputs = tokenize_question(question, tokenizer, model.device)

    if deterministic:
        generated_ids = model.generate(**model_inputs, max_new_tokens=512, do_sample=False, num_beams=1,
                                       temperature=None, top_p=None, top_k=None)
    else:
        generated_ids = model.generate(**model_inputs, max_new_tokens=512)
    generated_ids = [ output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids) ]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


def dataset_generator(dataset_, start_batch: int=0, batch_size: int=32) -> dict[str, Tensor]:
    """
    Generator to yield preprocessed batches from the dataset.
    Yields:
        dict: A dictionary containing the batch of data.
    """
    for i in range(start_batch, len(dataset_), batch_size):
        batch = dataset_.select(range(i, min(i + batch_size, len(dataset_))))
        yield i, { "input_ids": batch["input_ids"], "labels": batch["labels"] }


def tokenize_dataset(llm_dataset: Dataset, tokenizer: AutoTokenizer, max_length: int=512) -> Dataset:

    def preprocess_function(row):
        return tokenizer(row["text"], truncation=True, padding="max_length", max_length=max_length)

    llm_dataset = llm_dataset.map(preprocess_function)
    llm_dataset.set_format(type="torch", columns=["text", "input_ids", "attention_mask"])
    return llm_dataset
