import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AdamW, TrOCRProcessor, VisionEncoderDecoderModel, get_scheduler

from .configs import constants
from .context import Context
from .util import debug_print


def predict(
    processor: TrOCRProcessor, model: VisionEncoderDecoderModel, dataloader: DataLoader
) -> tuple[list[tuple[int, str]], list[float]]:
    output: list[tuple[int, str]] = []
    confidence_scores: list[tuple[int, float]] = []

    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(dataloader):
            debug_print(f"Predicting batch {i+1}")
            inputs: torch.Tensor = batch["input"].to(constants.device)

            generated_ids = model.generate(inputs, return_dict_in_generate=True, output_scores = True)
            generated_text = processor.batch_decode(generated_ids.sequences, skip_special_tokens=True)

            ids = [t.item() for t in batch["idx"]]
            output.extend(zip(ids, generated_text))

            # Compute confidence scores
            batch_confidence_scores = get_confidence_scores(generated_ids)
            confidence_scores.extend(zip(ids, batch_confidence_scores))

    return output, confidence_scores


def get_confidence_scores(generated_ids) -> list[float]:
    # Get raw logits, with shape (examples,tokens,token_vals)
    logits = generated_ids.scores
    logits = torch.stack(list(logits),dim=1)

    # Transform logits to softmax and keep only the highest (chosen) p for each token
    logit_probs = F.softmax(logits, dim=2)
    char_probs = logit_probs.max(dim=2)[0]

    # Only tokens of val>2 should influence the confidence. Thus, set probabilities to 1 for tokens 0-2
    mask = generated_ids.sequences[:,:-1] > 2
    char_probs[mask] = 1

    # Confidence of each example is cumulative product of token probs
    batch_confidence_scores = char_probs.cumprod(dim=1)[:, -1]
    return [v.item() for v in batch_confidence_scores]


# will return the accuracy but not print predictions
def validate(context: Context, print_wrong: bool = False) -> float:
    predictions, _ = predict(context.processor, context.model, context.val_dataloader)
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
    optimizer = AdamW(model.parameters(), lr=constants.learning_rate)

    num_training_steps = num_epochs * len(context.train_dataloader)
    lr_scheduler = get_scheduler(
        "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    model.to(constants.device)
    model.train()

    for epoch in range(num_epochs):
        for j, batch in enumerate(context.train_dataloader):
            inputs: torch.Tensor = batch["input"].to(constants.device)
            labels: torch.Tensor = batch["label"].to(constants.device)

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
