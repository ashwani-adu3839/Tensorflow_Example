from dataclasses import dataclass, field
from typing import Optional
import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    prev_data: int = field(
        default=32,
        metadata={
            "help": "tokens to be used to add reset data for next training sample"
        },
    )
    pad_token: int = field(
        default=-100,
        metadata={
            "help": "defaulty value in pytorch for loss function evaluation"
                },
    )
    device: str = field(
        default=device,
        metadata={
            "help": "device option CPU/GPU"
        },
    )
    INIT_LR: float = field(
        default=7e-5,
        metadata={
            "help": "initial learning rate"
        },
    )
    epochs : int = field(
        default=20,
        metadata={
            "help": "epoch"
        },
    )
    BS : int = field(
        default=2,
        metadata={
            "help": "batch size, make it a multiple of 8"
        },
    )
    bert_pretrained : str = field(
        default="../Model",
        metadata={
            "help": "pretrained lilth model weights path"
        },
    )
    fine_tuned_weights : str = field(
        default='model_weights/model.pt',
        metadata={
            "help": "fined tune model weights path"
        },
    )
    pickle_files : str = field(
        default='../data/labeled_data',
        metadata={
            "help": "path for training and testing dataset"
        },
    )
    data_folder : str = field(
        default='../data/train_data',
        metadata={
            "help": "path for preprocessed pickle files"
        },
    )
    tokenizer_path : str = field(
        default='../model_weights',
        metadata={
            "help": "path for tokenizer"
        },
    )
