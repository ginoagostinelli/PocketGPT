import os
import torch
import argparse
from gpt import GPT, ModelArgs
from utils import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_dir = os.path.join(os.path.dirname(__file__), "output/model.pt")
max_new_tokens = 100


@torch.no_grad()
def sample_sequence(model, tokenizer, prompt: str = "", max_new_tokens: int = 100):
    try:
        encoded_input = tokenizer(prompt, return_tensors="pt").to(device)
        inputs_ids = encoded_input["input_ids"]
        generated_tokens = model.generate(inputs_ids, max_new_tokens)
        generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        return generated_text
    except Exception as e:
        raise Exception(f"Error generating text: {e}")


def load_model(model_dir, device):
    args = ModelArgs()
    model = GPT(args)
    model.from_pretrained(model_dir)
    model.eval()
    model.to(device)
    return model


def main():
    parser = argparse.ArgumentParser(description="Generate text using a GPT model.")
    parser.add_argument("--prompt", type=str, help="The input prompt for text generation.")
    args = parser.parse_args()
    if args.prompt is not None:
        prompt = args.prompt
    else:
        prompt = "\n"

    model = load_model(model_dir, device)
    tokenizer = utils.get_tokenizer()
    sequence = sample_sequence(model, tokenizer, prompt, max_new_tokens)
    print(f"{sequence}")


if __name__ == "__main__":
    main()
