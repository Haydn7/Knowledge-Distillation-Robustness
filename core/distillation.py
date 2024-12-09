from typing import Any
from torch.optim import AdamW
import torch
from torch import nn
from torch.nn import functional as F
from itertools import islice
from transformers import AutoTokenizer
from utils.checkpointer import Checkpointer


def train_knowledge_distillation(teacher: nn.Module, student: nn.Module, dataloader: Any, tokenizer: AutoTokenizer,
                                 checkpointer: Checkpointer, lr: float=5.e-5, epochs: int=3, alpha: float=0.01) -> None:

    for param in teacher.parameters():
        param.requires_grad = False

    start_epoch, start_batch, total_loss = checkpointer.restore(student)

    optimizer = AdamW(student.parameters(), lr=lr)
    n_batches = len(dataloader)

    # Put the teacher in evaluation mode and the student in training mode
    teacher.eval()
    student.train()

    print(f"Running epochs between {start_epoch} and {epochs}")
    for epoch in range(start_epoch, epochs):
        data_iterator = islice(dataloader, start_batch, None)
        total_loss = 0.0
        for batch_idx, batch in enumerate(data_iterator, start=start_batch):
            input_ids = batch["input_ids"].to("cuda")
            attention_mask = batch["attention_mask"].to("cuda")

            # Teacher and student outputs
            with torch.no_grad():
                teacher_outputs = teacher(input_ids, attention_mask=attention_mask)
                teacher_logits = teacher_outputs.logits.to(torch.float32)

            student_outputs = student(input_ids, attention_mask=attention_mask)
            student_logits = student_outputs.logits.to(torch.float32)

            # Compute distillation loss (KL divergence)
            if True:
                distillation_loss = F.kl_div(
                    F.log_softmax(student_logits, dim=-1),
                    F.softmax(teacher_logits, dim=-1),
                    reduction="batchmean"
                )

            # Compute standard cross-entropy loss
            if True:
                labels = input_ids[:, 1:].contiguous()  # Shift labels for next-token prediction
                student_logits_shifted = student_logits[:, :-1, :].contiguous()
                cross_entropy_loss = F.cross_entropy(
                    student_logits_shifted.view(-1, student_logits.size(-1)),
                    labels.view(-1),
                    ignore_index=tokenizer.pad_token_id
                )
                loss = alpha * distillation_loss + (1 - alpha) * cross_entropy_loss

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if batch_idx % 64 == 0: 
                print(f"Epoch {epoch} batch {batch_idx} out of {n_batches} loss {total_loss / 64.0}")
                total_loss = 0.0

            if checkpointer.save_on_this_batch(batch_idx):
                checkpointer.save(student, epoch, batch_idx, loss.item())
        start_batch = 0
