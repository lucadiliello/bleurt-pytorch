import os
from typing import Union

from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from bleurt_pytorch.bleurt.tokenization_bleurt_fast import BleurtTokenizerFast
from bleurt_pytorch.bleurt.tokenization_bleurt_sp import BleurtSPTokenizer


class BleurtTokenizer(PreTrainedTokenizerBase):
    r"""
    Construct a BLEURT tokenizer and returns either `BleurtTokenizerFast` or `BleurtSPTokenizer` based
    on whether the model requires sentencepiece tokenization or not.
    """

    def __new__(cls, *args, **kwargs) -> Union[BleurtTokenizerFast, BleurtSPTokenizer]:
        try:
            return BleurtTokenizerFast(*args, **kwargs)
        except (OSError, TypeError):
            return BleurtSPTokenizer(*args, **kwargs)

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], *init_inputs, **kwargs
    ) -> Union[BleurtTokenizerFast, BleurtSPTokenizer]:
        try:
            return BleurtTokenizerFast.from_pretrained(pretrained_model_name_or_path, *init_inputs, **kwargs)
        except (OSError, TypeError):
            return BleurtSPTokenizer.from_pretrained(pretrained_model_name_or_path, *init_inputs, **kwargs)
