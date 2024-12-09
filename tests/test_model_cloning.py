import unittest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch import nn
from torch.nn import functional as F
import copy
from typing import Optional
from utils.tools import parameter_count, parameter_item_count
from utils.llm_tools import question_model
from core.replacement import replace_modules


class TestLinear(nn.Module):
    __constants__ = ["in_features", "out_features"]

    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(self, teacher: nn.Module, param_dtype: Optional[torch.dtype]=None) -> None:
        super().__init__()
        self.in_features, self.out_features = teacher.in_features, teacher.out_features
        dtype = param_dtype if param_dtype is not None else teacher.weight.dtype
        self.weight = nn.Parameter(teacher.weight.to(dtype).clone())

        if teacher.bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(teacher.bias.to(torch.float32).clone())

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == self.weight.dtype:
            return F.linear(x, self.weight, self.bias)
        else:
            x_type = x.dtype
            return F.linear(x.to(self.weight.dtype), self.weight, self.bias).to(x_type)


class TestReplacingModules(unittest.TestCase):
    """Tests the cloned model, which replaces int8 linear layers with float32 linear layers, has the correct number
    of parameters and returns correct answers according to the teacher."""

    @classmethod
    def setUpClass(cls):
        teacher_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        cls.teacher = AutoModelForCausalLM.from_pretrained(teacher_model_name, torch_dtype="auto", device_map="auto")
        print(cls.teacher)
        cls.tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.teacher.to(device)

        def valid_function(_, module, model):
            return isinstance(module, nn.Linear) and not module.weight is model.get_input_embeddings().weight

        cls.student = copy.deepcopy(cls.teacher)
        replace_modules(cls.student, TestLinear, valid_function, param_type=torch.float32)

    def test_cloned_parameter_item_count(self):
        self.assertEqual(parameter_item_count(self.teacher), parameter_item_count(self.student), "Parameter item count")

    def test_cloned_parameter_count(self):
        self.assertEqual(parameter_count(self.teacher), parameter_count(self.student), "Parameter count")

    def test_student_answer(self):

        question = "A board game spinner is divided into three parts labeled $A$, $B$ and $C$. The probability of the spinner landing on $A$ is $\frac{1}{3}$ and the probability of the spinner landing on $B$ is $\frac{5}{12}$.  What is the probability of the spinner landing on $C$? Express your answer as a common fraction."
        student_prompt = "Below is an instruction that describes a task. Write a concise response that appropriately completes the instruction."
        teacher_prompt = "You are a teacher marking the response of a student. Below is an instruction that describes a task. You can only reply CORRECT if you think the answer correctly answers the question or WRONG if it does not."

        student_question = f"{student_prompt}\n\n### Instruction:\n{question}\n\n### Response:\n"
        student_answer = question_model(self.student, self.tokenizer, student_question, deterministic=False)

        teacher_question = f"{teacher_prompt}\n\n### Instruction:\n{question}\n\n#### Student Answer:\n{student_answer}\n\n### Response:\n"
        teacher_grade = question_model(self.teacher, self.tokenizer, teacher_question, deterministic=True)
        teacher_grade = teacher_grade.replace(".", "").strip()
        self.assertEqual(teacher_grade, "CORRECT", "Teacher grade of answer")

if __name__ == '__main__':
    unittest.main()
