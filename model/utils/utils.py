import os
import torch
from transformers import GPT2Tokenizer

class PreTrainedModel:

    def from_pretrained(self, model_path: str) -> None:
        ''' Load the trained model '''
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'model not found in "{model_path}"')
        
        self.load_state_dict(torch.load(model_path))

def get_tokenizer():

    return GPT2Tokenizer.from_pretrained('gpt2')