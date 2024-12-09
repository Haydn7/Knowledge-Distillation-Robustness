from functools import partial
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer, QuantoConfig
import torch
from torch import nn
import copy
import os
from typing import Optional, Final
from torch.utils.data import DataLoader
import argparse
from data_loaders.rup_bench import llm_robust_scores

from utils.checkpointer import Checkpointer
from data_loaders.platypus import PlatypusData, get_platypus_dataset
from utils.tools import calculate_model_size, parameter_count, parameter_item_count
from utils.llm_tools import question_model
from core.replacement import  replace_modules, train_replacement_modules_on_teacher_modules
from core.distillation import train_knowledge_distillation
from core.group_lora_linear import GroupLoRALinear


STUDENT_MODEL_PATH: Final[str] = os.path.join(Path.home(), "data/models/QWen_distilled")
TEACHER_MODEL_NAME: Final[str] = "Qwen/Qwen2.5-0.5B-Instruct"
#TEACHER_MODEL_NAME: Final[str] = "Qwen/Qwen2.5-1.5B-Instruct"


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Investigates LLM sparsity and robustness.")

    parser.add_argument("-T", "--test", help="Test model on single question", action="store_true")
    parser.add_argument("-R", "--robust", help="Run robust evaluation", action="store_true")
    parser.add_argument("-L", "--layer", help="Run replacement layer training", action="store_true")
    parser.add_argument("-K", "--knowledge_distillation", help="Run knowledge distillation training", action="store_true")
    parser.add_argument("-S", "--only_test_teacher", help="Only test the teacher", action="store_true")
    parser.add_argument("-C", "--checkpoint", type=int, default=None, help="Load the specified checkpoint or take most recent")
    parser.add_argument("-E", "--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("-B", "--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("-W", "--weights", type=str, default="float16", help="Weights float type")
    return parser.parse_args()


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_parameter_details(name: str, model: nn.Module) -> None:
    model_size = calculate_model_size(model, "MB")
    print(f"{name} model size: {model_size:.1f}MB, parameter count: {parameter_count(model):,},",
          f"parameter item count: {parameter_item_count(model)}")


def create_teacher() -> nn.Module:
    teacher = AutoModelForCausalLM.from_pretrained(TEACHER_MODEL_NAME, torch_dtype="auto", device_map="auto")
    teacher.to(get_device())
    print_parameter_details("Teacher", teacher)
    return teacher


def create_student(teacher: nn.Module) -> nn.Module:
    student = copy.deepcopy(teacher)
    new_linear_model = partial(GroupLoRALinear, block_size=256, rank=256)

    def filter_function(_, module, model):
        return isinstance(module, nn.Linear) and not module.weight is model.get_input_embeddings().weight \
            and min(module.in_features, module.out_features) >= 256

    replace_modules(student, new_linear_model, filter_function, param_type=torch.float32)
    print_parameter_details("Student", student)
    return student


def test_models(args: argparse.Namespace,
                teacher: Optional[nn.Module] = None,
                student: Optional[nn.Module] = None,
                tokenizer : Optional[AutoTokenizer] = None) -> None:

    if teacher is None:
        teacher = create_teacher()
    if student is None and not args.only_test_teacher:
        student = create_student(teacher)
        checkpointer = Checkpointer(STUDENT_MODEL_PATH)
        checkpointer.restore(student, checkpoint_id=args.checkpoint)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_NAME)
    platypus = PlatypusData()
    dataset = platypus.load_dataset()

    prompt = platypus.question_prompt(dataset['train'][0])
    print(prompt)
    teacher_answer = question_model(teacher, tokenizer, prompt, deterministic=False)
    print("Teacher response:", teacher_answer)
    if not args.only_test_teacher:
        student_answer = question_model(student, tokenizer, prompt, deterministic=True)
        print("Student response:", student_answer)


def run_training(args):
    teacher = create_teacher()
    student = create_student(teacher)
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_NAME)

    dataset = get_platypus_dataset(tokenizer)
    train_dataloader = DataLoader(dataset["train"], shuffle=False, batch_size=1)
    checkpointer = Checkpointer(STUDENT_MODEL_PATH, batch_save_step=64)

    if args.replace:
        train_replacement_modules_on_teacher_modules(teacher, student, train_dataloader, lambda obj: isinstance(obj, GroupLoRALinear),
                                                     checkpointer, epochs=args.epochs, lr=4.e-3)
    if args.knowledge_distillation:
        epochs = 2 * args.epochs if args.replaec else args.epochs
        train_knowledge_distillation(teacher, student, train_dataloader, tokenizer, checkpointer, lr=1.e-5, epochs=epochs)

    if args.test:
        test_models(args, teacher=teacher, student=student, tokenizer=tokenizer)

def run_robust_scores(args: argparse.Namespace) -> None:

    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_NAME)
    if args.weights in ["float8", "int8", "int4", "int2"]:
        quantization_config = QuantoConfig(weights=args.weights)
        model = AutoModelForCausalLM.from_pretrained(TEACHER_MODEL_NAME, quantization_config=quantization_config)
    else:
        model = AutoModelForCausalLM.from_pretrained(TEACHER_MODEL_NAME, torch_dtype="auto", device_map="auto")
    model.to(get_device())
    print(model)
    print_parameter_details(TEACHER_MODEL_NAME, model)
    llm_robust_scores(model, tokenizer)


if __name__ == "__main__":
    args = get_arguments()
    if args.robust:
        run_robust_scores(args)
    elif args.layer or args.knowledge_distillation:
        run_training(args)
    elif args.test:
        test_models(args)

