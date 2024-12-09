from transformers import AutoTokenizer
import pandas as pd
from typing import Any, Final
from torch import nn, Tensor
import numpy as np
from utils.csv_dataset import CSVDataset

RIDDLE_FILE: Final[str] = "data/formated_RiddleSense_RUP.csv"
COMMON_SENSE_FILE: Final[str] = "data/formated_CommonsenseQA_RUP.csv"


def tokenize_questions(row: Any, tokenizer: AutoTokenizer, perturbations: list[str]) -> dict[str, Tensor]:
    system_prompt = "Your task is to answer the following multiple-choice question accurately and efficiently. " \
                    + f"Only answer with the corresponding letter from [{row['letters']}] for the correct option. " \
                    + "Do not show any reasoning.\n\nQuestion:\n"

    messages = [[{"role": "user", "content": f"{system_prompt}{row[p]}\n\nOptions:\n{row['options']}" }] for p in perturbations]
    prompts = tokenizer.apply_chat_template( messages, tokenize=False, add_generation_prompt=True )
    return tokenizer(prompts, return_tensors="pt", truncation=True, padding="max_length", padding_side="left", max_length=512)


def llm_robust_scores(model: nn.Module, tokenizer: AutoTokenizer) -> dict[str, float]:

    answer_column = "answerKey"
    perturbations = ['question', 'question_stem_red_herrings', 'question_stem_typos', 'question_stem_HP',
                     'question_stem_Leet', 'question_stem_It_cleft', 'question_stem_Wh_cleft', 'question_stem_stress',
                     'question_stem_checklist', 'question_stem_comp']

    dataset = CSVDataset(RIDDLE_FILE)
    model.eval()

    total_scores = np.zeros(len(perturbations))
    for batch in range(len(dataset)):
        row = dataset[batch]

        # Generate a batch containing all possible questions
        tokens = tokenize_questions(row, tokenizer, perturbations)
        inputs = tokens["input_ids"].to(model.device)
        attention_mask = tokens["attention_mask"].to(model.device)
        output_ids = model.generate(inputs, attention_mask=attention_mask, max_new_tokens=8, pad_token_id=tokenizer.eos_token_id)

        # Decode the generated tokens, removing the inputs
        output_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(tokens.input_ids, output_ids)]
        response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        question_score = np.array([1.0 if row[answer_column] in r else 0.0 for r in response])
        total_scores += question_score

        if batch % 32 == 0:
            print(f"batch {batch} accuracy {100.0 * total_scores[0] / (batch + 1):.4}% response {response[0]} target {row[answer_column]} scores {total_scores / (batch+1)}")
    accuracy = total_scores / len(dataset)
    print("Accuracy", accuracy)
    return perturbations, accuracy


def format_csv_file(source_file: str, destination_file: str) -> None:
    """ Corrects the format of data in the csv file and abstracts the options"""

    # Abstract options from the dataset
    def abstract_options(question: str) -> str:
        try:
            js = eval(question)
            return "\n".join([f" Option {r['label']}: {r['text']}." for r in js['choices']])
        except Exception as e:
            print("Unable to decode", question, "ERROR", e)

    def abstract_letters(question: str) -> str:
        try:
            js = eval(question)
            return ",".join([r['label'] for r in js['choices']])
        except Exception as e:
            print("Unable to decode", question, "ERROR", e)

    df = pd.read_csv(source_file)  # Load the CSV file into a DataFrame
    df["letters"] = df["question"].apply(abstract_letters)
    df["options"] = df["question"].apply(abstract_options)
    df["question"] = df["question_stem"]
    df.drop(columns=["question_stem"])
    df.to_csv(destination_file)
    print(df.iloc[0]["options"])


if __name__ == "__main__":
    format_csv_file("../data/CommonsenseQA_RUP.csv", "../" + COMMON_SENSE_FILE)

