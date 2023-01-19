# bleurt-pytorch

Use BLEURT models in native PyTorch with [Transformers](https://huggingface.co/transformers).

## Getting started

Install with:

```bash
pip install git+https://github.com/lucadiliello/bleurt-pytorch.git
```

Now load your favourite model with:

```python
import torch
from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer

config = BleurtConfig.from_pretrained('lucadiliello/BLEURT-20-D12')
model = BleurtForSequenceClassification.from_pretrained('lucadiliello/BLEURT-20-D12')
tokenizer = BleurtTokenizer.from_pretrained('lucadiliello/BLEURT-20-D12')

references = ["a bird chirps by the window", "this is a random sentence"]
candidates = ["a bird chirps by the window", "this looks like a random sentence"]

model.eval()
with torch.no_grad():
    inputs = tokenizer(references, candidates, padding='longest', return_tensors='pt')
    res = model(**inputs).logits.flatten().tolist()
print(res)
# [0.9604414105415344, 0.8080050349235535]
```

You can find all BLUERT models adapted for PyTorch [here](https://huggingface.co/lucadiliello). The recommended model is `lucadiliello/BLEURT-20`, however this model is very large and may require too much resources. `BLEURT-20-D12` is smaller but works well enough for most comparisons.


## Credits

- [Google original BLEURT](https://github.com/google-research/bleurt) implementation
- [Transformers](https://huggingface.co/transformers) project
- Users of this [issue](https://github.com/huggingface/datasets/issues/224), from which I took inspiration.
