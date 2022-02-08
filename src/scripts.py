import torch
from torch.utils.data import DataLoader
from transformers import AdamW, TrOCRProcessor, VisionEncoderDecoderModel, get_scheduler

from .configs.constants import device
from .context import Context
from .util import debug_print


def predict(
    processor: TrOCRProcessor, model: VisionEncoderDecoderModel, dataloader: DataLoader
) -> list[tuple[int, str]]:
    output: dict[int, str] = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            debug_print(f"Predicting batch {i+1}")
            inputs: torch.Tensor = batch["input"].to(device)

            generated_ids = model.generate(inputs)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)

            ids = [t.item() for t in batch["idx"]]
            output.extend(zip(ids, generated_text))

    return output


# will return the accuracy but not print predictions
def validate(context: Context, print_wrong: bool = False) -> float:
    predictions = predict(context.processor, context.model, context.val_dataloader)
    assert len(predictions) > 0

    correct_count = 0
    wrong_count = 0
    for id, prediction in predictions:
        label = context.val_dataset.get_label(id)
        path = context.val_dataset.get_path(id)

        if prediction == label:
            correct_count += 1
        else:
            wrong_count += 1
            if print_wrong:
                print(f"Predicted: \t{prediction}\nLabel: \t\t{label}\nPath: \t\t{path}")

    if print_wrong:
        print(f"\nCorrect: {correct_count}\nWrong: {wrong_count}")
    return correct_count / (len(predictions))


def train(context: Context, num_epochs=5):
    model = context.model
    optimizer = AdamW(model.parameters(), lr=5e-5)

    num_training_steps = num_epochs * len(context.train_dataloader)
    lr_scheduler = get_scheduler(
        "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        for j, batch in enumerate(context.train_dataloader):
            inputs: torch.Tensor = batch["input"].to(device)
            labels: torch.Tensor = batch["label"].to(device)

            outputs = model(pixel_values=inputs, labels=labels)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            debug_print(f"Epoch {1 + epoch}, Batch {1 + j}: {loss} loss")
            del loss, outputs

        if len(context.val_dataloader) > 0:
            accuracy = validate(context)
            print(f"\n---- Epoch {1 + epoch} ----\nAccuracy: {accuracy}\n\n")
