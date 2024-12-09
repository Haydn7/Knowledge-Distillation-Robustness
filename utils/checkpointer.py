import torch
import os
import re
from typing import Optional


class Checkpointer:
    def __init__(self, directory, batch_save_step = 128):
        self.root_directory, self.batch_save_step = directory, batch_save_step
        self.replacement_directory = os.path.join(directory, "replacement")
        os.makedirs(self.replacement_directory, exist_ok=True)
        pattern = re.compile(r'_(\d+)\.pt$')
        matches = [pattern.search(f) for f in os.listdir(self.replacement_directory)]
        self.next_checkpoint_id = max([0] + [1 + int(m.group(1)) for m in matches if m])

    def save_on_this_batch(self, batch: int) -> bool:
        return batch % self.batch_save_step == 0

    def file_path(self, count):
        return os.path.join(self.replacement_directory, f"student_checkpoint_{count}.pt")

    def save(self, model, epoch, batch, loss) -> None:
        """Saves a checkpoint of the model and training state."""
        checkpoint = {
            "model_state": model.state_dict(),
            "status" : (epoch, batch, loss),
        }
        file_path = self.file_path(self.next_checkpoint_id)
        print(f"Saving checkpoint for epoch {epoch} batch {batch} loss {loss} to {file_path}")
        torch.save(checkpoint, file_path)
        self.next_checkpoint_id += 1

    def restore(self, model: torch.nn.Module, file_path: Optional[str] = None, 
            checkpoint_id: Optional[int] = None) -> tuple[int, int, int]:
        """Loads a checkpoint and restores the model state."""
        if file_path is None:
            file_path = self.file_path(self.next_checkpoint_id - 1 if checkpoint_id is None else checkpoint_id)

        if os.path.exists(file_path):
            checkpoint = torch.load(file_path, weights_only=True)
            model.load_state_dict(checkpoint['model_state'])
            print(f"Restoring from {file_path} with status {checkpoint['status']}")
            return checkpoint["status"]
        return 0, 0, 0
