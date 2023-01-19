import logging
import os
from argparse import ArgumentParser, Namespace


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


import tensorflow.compat.v1 as tf  # noqa: E402
import torch  # noqa: E402
from bleurt import score as bleurt_score  # noqa: E402

from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer  # noqa: E402
from bleurt_pytorch.bleurt.tokenization_bleurt_fast import BleurtTokenizerFast  # noqa: E402
from bleurt_pytorch.bleurt.tokenization_bleurt_sp import BleurtSPTokenizer  # noqa: E402


references = ["a bird chirps by the window", "this is the first phd of Luca"]
candidates = ["a bird chirps by the window", "this looks the first phd of Luca"]


logging.basicConfig()
logger = logging.getLogger('bleurt_pytorch')
logger.setLevel(logging.INFO)


def build_pytorch_state_dict(tensorflow_model_path: str):
    r""" Build PT state dict from TF checkpoint. """

    imported = tf.saved_model.load_v2(tensorflow_model_path)
    state_dict = {}

    for variable in imported.variables:
        n: str = variable.name
        if n.startswith('global'):
            continue

        data = variable.numpy()

        if 'kernel' in n:  # this is fix #1 - considering 'kernel' layers instead of 'dense'
            data = data.T

        n = n.split(':')[0]
        n = n.replace('/', '.')
        n = n.replace('_', '.')

        if 'bert' in n:
            n = n.replace('bert', 'bleurt')

        n = n.replace('kernel', 'weight')

        if 'embedding.hidden.mapping.in' in n:
            n = n.replace('embedding.hidden.mapping.in', 'embedding_projection')
        elif 'LayerNorm' in n:
            n = n.replace('beta', 'bias')
            n = n.replace('gamma', 'weight')
        elif 'embeddings' in n:
            n = n.replace('word.embeddings', 'word_embeddings')
            n = n.replace('position.embeddings', 'position_embeddings')
            n = n.replace('token.type.embeddings', 'token_type_embeddings')
            n = n + '.weight'

        elif n.startswith('dense.weight'):
            n = n.replace('dense.weight', 'classifier.weight')
        elif n.startswith('dense.bias'):
            n = n.replace('dense.bias', 'classifier.bias')

        state_dict[n] = torch.from_numpy(data)

    return state_dict


def main(args: Namespace):

    logger.info(f"Processing folder {args.input}...")
    logger.info("Loading original TF model...")

    # original TF BLUERT implementation scores
    original_scorer = bleurt_score.BleurtScorer(args.input)
    logger.info("Computing original TF scores...")
    res = original_scorer.score(references=references, candidates=candidates)
    logger.info(f"TensorFlow model scores: {res}")

    logger.info("Loading PyTorch model...")

    # my PT implementation scores
    config = BleurtConfig.from_pretrained(
        os.path.join(args.input, 'bert_config.json'),
        num_labels=1,
        architectures=['BleurtForSequenceClassification'],
    )

    logger.info("Creating new state_dict from TF checkpoint...")
    state_dict = build_pytorch_state_dict(args.input)

    logger.info("Building model...")
    model = BleurtForSequenceClassification(config)
    incompatible_keys = model.load_state_dict(state_dict, strict=False)
    logger.debug(f"Missing keys: {incompatible_keys.missing_keys}")
    logger.debug(f"Unexpected keys: {incompatible_keys.unexpected_keys}")

    logger.info("Loading PT tokenizer")
    try:
        tok = BleurtTokenizerFast.from_pretrained(args.input)
    except OSError:
        tok = BleurtSPTokenizer(os.path.join(args.input, 'sent_piece.model'))

    logger.info("Computing PyTorch scores...")
    model.eval()
    with torch.no_grad():
        res = model(**tok(references, candidates, padding='longest', return_tensors='pt')).logits.flatten().numpy()
    logger.info(f"PyTorch model scores: {res}")

    config.save_pretrained(args.output)
    model.save_pretrained(args.output)
    tok.save_pretrained(args.output)

    logging.info("Reload check...")
    config = BleurtConfig.from_pretrained(args.output)
    model = BleurtForSequenceClassification.from_pretrained(args.output)
    tok = BleurtTokenizer.from_pretrained(args.output)

    with torch.no_grad():
        res = model(**tok(references, candidates, padding='longest', return_tensors='pt')).logits.flatten().numpy()
    logger.info(f"Final model scores: {res}")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
    main(args)
