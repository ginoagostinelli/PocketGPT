# PocketGPT

With PocketGPT you can easily train your own GPT models ;)

## 🔧  Install requirements
```python
pip install -r requirements.txt```

## Train your model!!
Download a dataset:
```python
python data/openwebtext/get_openwebtext_dataset.py
```
Train the model:
```python
python model/trainer.py
```

## Inference
```python
python model/sample.py --prompt="Here goes you prompt"
```