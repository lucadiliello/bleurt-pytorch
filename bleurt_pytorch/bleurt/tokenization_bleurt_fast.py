from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast


VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt", "tokenizer_file": "tokenizer.json"}


PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "lucadiliello/bleurt-tiny-128": "https://huggingface.co/lucadiliello/bleurt-tiny-128/resolve/main/vocab.txt",
        "lucadiliello/bleurt-tiny-512": "https://huggingface.co/lucadiliello/bleurt-tiny-512/resolve/main/vocab.txt",
        "lucadiliello/bleurt-base-128": "https://huggingface.co/lucadiliello/bleurt-base-128/resolve/main/vocab.txt",
        "lucadiliello/bleurt-base-512": "https://huggingface.co/lucadiliello/bleurt-base-512/resolve/main/vocab.txt",
        "lucadiliello/bleurt-large-128": "https://huggingface.co/lucadiliello/bleurt-large-128/resolve/main/vocab.txt",
        "lucadiliello/bleurt-large-512": "https://huggingface.co/lucadiliello/bleurt-large-512/resolve/main/vocab.txt",
    },
    "tokenizer_file": {
        "lucadiliello/bleurt-tiny-128": "https://huggingface.co/lucadiliello/bleurt-tiny-128/resolve/main/tokenizer.json",  # noqa: E501
        "lucadiliello/bleurt-tiny-512": "https://huggingface.co/lucadiliello/bleurt-tiny-512/resolve/main/tokenizer.json",  # noqa: E501
        "lucadiliello/bleurt-base-128": "https://huggingface.co/lucadiliello/bleurt-base-128/resolve/main/tokenizer.json",  # noqa: E501
        "lucadiliello/bleurt-base-512": "https://huggingface.co/lucadiliello/bleurt-base-512/resolve/main/tokenizer.json",  # noqa: E501
        "lucadiliello/bleurt-large-128": "https://huggingface.co/lucadiliello/bleurt-large-128/resolve/main/tokenizer.json",  # noqa: E501
        "lucadiliello/bleurt-large-512": "https://huggingface.co/lucadiliello/bleurt-large-512/resolve/main/tokenizer.json",  # noqa: E501
    },
}


PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "lucadiliello/bleurt-base-128": 128,
    "lucadiliello/bleurt-base-512": 512,
    "lucadiliello/bleurt-large-128": 128,
    "lucadiliello/bleurt-large-512": 512,
    "lucadiliello/bleurt-tiny-128": 128,
    "lucadiliello/bleurt-tiny-512": 512,
}


PRETRAINED_INIT_CONFIGURATION = {
    "lucadiliello/bleurt-base-128": {"do_lower_case": True},
    "lucadiliello/bleurt-base-512": {"do_lower_case": True},
    "lucadiliello/bleurt-large-128": {"do_lower_case": True},
    "lucadiliello/bleurt-large-512": {"do_lower_case": True},
    "lucadiliello/bleurt-tiny-128": {"do_lower_case": True},
    "lucadiliello/bleurt-tiny-512": {"do_lower_case": True},
}


class BleurtTokenizerFast(BertTokenizerFast):
    r"""
    Construct a "fast" BLEURT tokenizer (backed by HuggingFace's *tokenizers* library). Based on WordPiece.
    This class is identical to `BertTokenizerFast` apart from the default values for tokenizer files,
    initial configurations and positional embeddings defaults.

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            File containing the vocabulary.
        do_lower_case (`bool`, *optional*, defaults to `True`):
            Whether or not to lowercase the input when tokenizing.
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
        clean_text (`bool`, *optional*, defaults to `True`):
            Whether or not to clean the text before tokenization by removing any control characters and replacing all
            whitespaces by the classic one.
        tokenize_chinese_chars (`bool`, *optional*, defaults to `True`):
            Whether or not to tokenize Chinese characters. This should likely be deactivated for Japanese (see [this
            issue](https://github.com/huggingface/transformers/issues/328)).
        strip_accents (`bool`, *optional*):
            Whether or not to strip all accents. If this option is not specified, then it will be determined by the
            value for `lowercase` (as in the original BERT).
        wordpieces_prefix (`str`, *optional*, defaults to `"##"`):
            The prefix for subwords.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
