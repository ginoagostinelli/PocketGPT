from gpt import GPT
import os
import torch
import numpy as np
import torch.nn as nn

# config
batch_size = 32
batch_iterations = 2500
eval_interval = 220
learning_rate = 3e-4
predict_iters = 220
context_size = 12
save_checkpoint = True
# save_steps = 1
resume_from_checkpoint = False
dataset = 'cornell_movie_dialogs'
dataset_path = os.path.join('data', dataset)
# ------------
output_dir = os.path.join(os.path.dirname(__file__), 'output')

class Trainer:

    def __init__(self, model: nn.Module):
        # TODO: Check model parameters
        if model is None:
            raise RuntimeError('"Trainer" requires a model to be specified')
        self.model = model

        # TODO: add a data loader 
        self.train_dataset = np.memmap(os.path.join(dataset_path, 'train.bin'), dtype=np.uint16, mode='r')
        self.val_dataset = np.memmap(os.path.join(dataset_path, 'val.bin'), dtype=np.uint16, mode='r')


    def get_batch(self, split):
        data = self.train_dataset if split == 'train' else self.val_dataset
        ix = torch.randint(len(data) - context_size, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+context_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+context_size]).astype(np.int64)) for i in ix])
        return x, y
    

    @torch.no_grad()
    def predict_loss(self):
        output = {}
        self.model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(predict_iters)
            for k in range(predict_iters):
                X, Y = self.get_batch(split)
                loss, _ = self.model(X, Y)
                losses[k] = loss.item()
            output[split] = losses.mean()
        self.model.train()
        return output


    def save_model(self, output_dir: str) -> None:
        '''Save the trained model, then you can reload it with from_pretrained()'''
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(output_dir, 'model.pt'))


    def from_pretrained(self, output_dir: str) -> None:
        ''' Load the trained model '''
        if not os.path.exists(os.path.join(output_dir, 'model.pt')):
            raise FileNotFoundError(f'"model.pt" not found in "{output_dir}"')
        
        self.model.load_state_dict(torch.load(os.path.join(output_dir, 'model.pt')))


    def train(self):
        if resume_from_checkpoint:
            self.from_pretrained(output_dir)
    
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

        for iter in range(batch_iterations):

            if iter % eval_interval == 0 or iter == batch_iterations - 1:
                predicted_loss = self.predict_loss()
                print(f"STEP {iter} --> Training loss: {predicted_loss['train']:.4f} || Validation loss: {predicted_loss['val']:.4f}")

            xb, yb = self.get_batch('train')

            loss, _ = self.model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            
            loss.backward()
            optimizer.step()
        
        if save_checkpoint:
            self.save_model(output_dir)


def main():
    model = GPT()
    trainer = Trainer(model)
    trainer.train()


if __name__ == "__main__":    
    main()