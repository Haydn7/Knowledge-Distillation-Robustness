import torch
from torch import nn
from typing import Type, Optional, Callable, Generator
from utils.checkpointer import Checkpointer
from itertools import islice


def replace_modules(model: nn.Module, replacement_class: Type[nn.Module],
                    valid_function: Callable[[str, nn.Module, nn.Module], bool],
                    node: Optional[nn.Module] = None,
                    param_type: Optional[torch.dtype] = None) -> None:

    """
    Replace modules of a given type with a replacement class, skipping certain modules based on a condition.
    Args:
        :param model                    The model to modify
        :param replacement_class        The replacement module class (e.g., BlockLoRaLinear)
        :param valid_function           A function that accepts the module name and the module instance and return True if the module should be replaced
        :param node                     The current node, should be None if the current node is the root node
        :param param_type               If specified, casts the new module's parameters to this dtype
    """
    if node is None:
        node = model
    for name, child in node.named_children():
        if valid_function(name, child, model):
            device = next(child.parameters()).device if hasattr(child, 'parameters') else torch.device('cpu')
            setattr(node, name, replacement_class(child, param_type).to(device))
        else:
            replace_modules(model, replacement_class, valid_function, node=child, param_type=param_type)


# Function to replace Linear layers with BlockLoRaLinear and track replacements
def map_modules(model_a: nn.Module, model_b: nn.Module, b_filter: Callable[[nn.Module], bool]) -> dict[nn.Module, nn.Module]:
    module_map = {}
    for child_a, child_b in zip(model_a.children(), model_b.children()):
        if b_filter(child_b):
            module_map[child_a] = child_b
        else:
            module_map.update(map_modules(child_a, child_b, b_filter))
    return module_map



def filter_parameters_generator(model: nn.Module, valid_function: Callable[[nn.Module], bool]) -> Generator[
    nn.Parameter, None, None]:
    """
    Returns the parameters of the model where the child modules according to the valid_function

    :param model:               model
    :param valid_function:      function that takes the nn.Module and return True if the parameters should be returned
    :return:                    Generator of the filtered model parameters
    """
    for module in model.modules():
        if valid_function(module):
            yield from module.parameters()


def train_replacement_modules_on_teacher_modules(teacher: nn.Module, student: nn.Module, dataloader,
                                                 is_trainable_function: Callable[[nn.Module], bool],
                                                 checkpointer: Checkpointer, epochs: int=5, lr: float=0.001):

    # Set the teacher to evaluation mode, and the student to training mode
    teacher.eval()
    student.train()

    start_epoch, start_batch, total_loss = checkpointer.restore(student)

    captured_data = map_modules(teacher, student, is_trainable_function)
    captured_data = { t: { "student": s, "optimizer": torch.optim.Adam(s.parameters(), lr=lr) } for t, s in captured_data.items() }

    # Forward hook to capture input/output of the teacher
    def capture_input_output(module_, input_, output_):
        captured_data[module_]["inputs"] = input_[0].detach().clone()
        captured_data[module_]["outputs"] = output_.detach().clone()

    hooks = [t.register_forward_hook(capture_input_output) for t in captured_data]

    loss_function = torch.nn.MSELoss()
    n_rows = len(dataloader)
    n_modules = len(captured_data)

    data_iterator = islice(dataloader, start_batch, None)
    for epoch in range(start_epoch, epochs):
        for batch, batch_data in enumerate(data_iterator, start=start_batch):
            input_ids = batch_data["input_ids"].to(teacher.device)
            attention_mask = batch_data["attention_mask"].to(teacher.device)

            # Forward pass with the teacher then use the hooks to train the student on the input and outputs
            with torch.no_grad():
                teacher(input_ids, attention_mask=attention_mask)

            mean_loss = 0
            for data in captured_data.values():
                student_outputs = data["student"].forward(data["inputs"])
                loss = loss_function(student_outputs, data["outputs"])

                # Backpropagation
                data["optimizer"].zero_grad()
                loss.backward()
                data["optimizer"].step()

                mean_loss += loss.item()
            mean_loss /= n_modules
            print(f"Processing batch {batch} of {n_rows} loss {mean_loss}")
            total_loss += mean_loss

            if checkpointer.save_on_this_batch(batch):
                checkpointer.save(student, epoch, batch, total_loss)

        print(f"Epoch {epoch + 1}, Loss: {total_loss / n_rows}")
        total_loss, start_batch = 0.0, 0

    for hook in hooks:
        hook.remove()
