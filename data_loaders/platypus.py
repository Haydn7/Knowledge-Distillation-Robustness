from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
from utils.llm_tools import tokenize_dataset

# Load the Open-Platypus dataset

class PlatypusData:

    def __init__(self) -> None:
        self.q_prompt_str = "Below is an instruction that describes a task. Write a concise response that appropriately completes the instruction.\n\n### Instruction:\n{instruction}\n\n### Response:\n"
        self.qa_prompt_str = self.q_prompt_str + "\n{response}"

    @staticmethod
    def load_dataset() -> Dataset:
        return load_dataset("garage-bAInd/Open-Platypus")

    def question_answer_prompt(self, row: dict[str, str]) -> dict[str, str]:
        return { "text" : self.qa_prompt_str.format(instruction=row["instruction"], response=row["output"])}

    def question_prompt(self, row):
        return self.q_prompt_str.format(instruction=row["instruction"])

    @staticmethod
    def parse_response(response: str) -> str:
        return response.split("### Response:")[1].strip()


def get_platypus_dataset(tokenizer: AutoTokenizer) -> Dataset:
    platypus = PlatypusData()
    dataset = platypus.load_dataset()
    dataset = dataset.map(lambda row : platypus.question_answer_prompt(row))
    return tokenize_dataset(dataset, tokenizer, max_length=1024)
