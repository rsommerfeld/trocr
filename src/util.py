from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from .configs import paths
from .configs import constants


def load_processor() -> TrOCRProcessor:
    return TrOCRProcessor.from_pretrained(paths.trocr_repo)


def load_model(from_disk: bool) -> VisionEncoderDecoderModel:
    if from_disk:
        assert paths.model_path.exists(), f"No model existing at {paths.model_path}"
        model: VisionEncoderDecoderModel = VisionEncoderDecoderModel.from_pretrained(paths.model_path)
        debug_print(f"Loaded local model from {paths.model_path}")
    else:
        model: VisionEncoderDecoderModel = VisionEncoderDecoderModel.from_pretrained(paths.trocr_repo)
        debug_print(f"Loaded pretrained model from huggingface ({paths.trocr_repo})")

    debug_print(f"Using device {constants.device}.")
    model.to(constants.device)
    return model


def init_model_for_training(model: VisionEncoderDecoderModel, processor: TrOCRProcessor):
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size


def debug_print(string: str):
    if constants.should_log:
        print(string)
