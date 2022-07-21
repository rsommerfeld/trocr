import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

NUM_WORKERS = 8
SHOULD_LOG = True

BATCH_SIZE = 20
TRAIN_EPOCHS = 8
WORD_LEN_PADDING = 8  # will be overriden if the dataset contains labels longer than the constant
LEARNING_RATE = 5e-6