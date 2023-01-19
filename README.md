# bleurt-pytorch

Use BLEURT models in Native PyTorch with [Transformers](https://huggingface.co/transformers).

## Getting started

Install with:

```bash
pip install git+https://github.com/lucadiliello/bleurt-pytorch.git
```

Now load your favourite model with:

```python
import torch
from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer

config = BleurtConfig.from_pretrained('lucadiliello/BLEURT-20')
model = BleurtForSequenceClassification.from_pretrained('lucadiliello/BLEURT-20')
tokenizer = BleurtTokenizer.from_pretrained('lucadiliello/BLEURT-20')

references = ["a bird chirps by the window", "this is a random sentence"]
candidates = ["a bird chirps by the window", "this looks like a random sentence"]

model.eval()
with torch.no_grad():
    res = model(**tok(references, candidates, padding='longest', return_tensors='pt')).logits.flatten().numpy()
```

You can find all BLUERT models adapted for PyTorch [here](https://huggingface.co/lucadiliello).
