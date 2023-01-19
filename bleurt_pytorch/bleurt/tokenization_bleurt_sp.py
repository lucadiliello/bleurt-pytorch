from transformers.models.deberta_v2.tokenization_deberta_v2 import DebertaV2Tokenizer


PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "lucadiliello/BLEURT-20-D3": "https://huggingface.co/lucadiliello/BLEURT-20-D3/resolve/main/spm.model",
        "lucadiliello/BLEURT-20-D6": "https://huggingface.co/lucadiliello/BLEURT-20-D6/resolve/main/spm.model",
        "lucadiliello/BLEURT-20-D12": "https://huggingface.co/lucadiliello/BLEURT-20-D12/resolve/main/spm.model",
        "lucadiliello/BLEURT-20": "https://huggingface.co/lucadiliello/BLEURT-20/resolve/main/spm.model",

    }
}


PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "lucadiliello/BLEURT-20-D3": 512,
    "lucadiliello/BLEURT-20-D6": 512,
    "lucadiliello/BLEURT-20-D12": 512,
    "lucadiliello/BLEURT-20": 512,
}


PRETRAINED_INIT_CONFIGURATION = {
    "lucadiliello/BLEURT-20-D3": {"do_lower_case": False},
    "lucadiliello/BLEURT-20-D6": {"do_lower_case": False},
    "lucadiliello/BLEURT-20-D12": {"do_lower_case": False},
    "lucadiliello/BLEURT-20": {"do_lower_case": False},
}


VOCAB_FILES_NAMES = {"vocab_file": "spm.model"}


class BleurtSPTokenizer(DebertaV2Tokenizer):
    r"""
    Constructs a BLEURT tokenizer based on [SentencePiece](https://github.com/google/sentencepiece). The code inherits
    from [DebertaV2Tokenizer](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deberta_v2/tokenization_deberta_v2.py) # noqa: E501
    because it is the most similar SP tokenizer that is ready-to-be-used with Google's BERT ckpts.

    Args:
        vocab_file (`str`):
            [SentencePiece](https://github.com/google/sentencepiece) file (generally has a *.spm* extension) that
            contains the vocabulary necessary to instantiate a tokenizer.
        do_lower_case (`bool`, *optional*, defaults to `False`):
            Whether or not to lowercase the input when tokenizing.
        bos_token (`string`, *optional*, defaults to `"[CLS]"`):
            The beginning of sequence token that was used during pre-training. Can be used a sequence classifier token.
            When building a sequence using special tokens, this is not the token that is used for the beginning of
            sequence. The token used is the `cls_token`.
        eos_token (`string`, *optional*, defaults to `"[SEP]"`):
            The end of sequence token. When building a sequence using special tokens, this is not the token that is
            used for the end of sequence. The token used is the `sep_token`.
        unk_token (`str`, *optional*, defaults to `"[UNK]"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        sep_token (`str`, *optional*, defaults to `"[SEP]"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        pad_token (`str`, *optional*, defaults to `"[PAD]"`):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token (`str`, *optional*, defaults to `"[CLS]"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        mask_token (`str`, *optional*, defaults to `"[MASK]"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        sp_model_kwargs (`dict`, *optional*):
            Will be passed to the `SentencePieceProcessor.__init__()` method. The [Python wrapper for
            SentencePiece](https://github.com/google/sentencepiece/tree/master/python) can be used, among other things,
            to set:

            - `enable_sampling`: Enable subword regularization.
            - `nbest_size`: Sampling parameters for unigram. Invalid for BPE-Dropout.

              - `nbest_size = {0,1}`: No sampling is performed.
              - `nbest_size > 1`: samples from the nbest_size results.
              - `nbest_size < 0`: assuming that nbest_size is infinite and samples from the all hypothesis (lattice)
                using forward-filtering-and-backward-sampling algorithm.

            - `alpha`: Smoothing parameter for unigram sampling, and dropout probability of merge operations for
              BPE-dropout.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
