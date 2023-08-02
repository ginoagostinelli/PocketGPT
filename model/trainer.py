from gpt import GPT, ModelArgs
import os
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

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
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# ------------
output_dir = os.path.join(os.path.dirname(__file__), 'output')

class Dataset(Dataset):
    def __init__(self, dataset_path):
        self.dataset = np.memmap(dataset_path, dtype=np.uint16, mode='r')

    def __len__(self):
        return len(self.dataset) - context_size

    def __getitem__(self, idx):
        x = torch.from_numpy(self.dataset[idx: idx + context_size].astype(np.int64))
        y = torch.from_numpy(self.dataset[idx + 1: idx + 1 + context_size].astype(np.int64))
        return x, y


class Trainer:

    def __init__(self, model: nn.Module):
        # TODO: Check model parameters
        if model is None:
            raise RuntimeError('"Trainer" requires a model to be specified')
        self.model = model

        self.train_dataset = Dataset(os.path.join(dataset_path, 'train.bin'))
        self.val_dataset = Dataset(os.path.join(dataset_path, 'val.bin'))

    @staticmethod
    def get_dataloader(dataset, shuffle=True) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=(device == "cuda"),
        )

    @torch.no_grad()
    def predict_loss(self) -> dict:
        output = {}
        self.model.eval()

        for split, dataloader in [("train", self.get_dataloader(self.train_dataset, shuffle=False)),
                                   ("val", self.get_dataloader(self.val_dataset, shuffle=False))]:
            data = iter(dataloader)
            losses = torch.zeros(predict_iters)
            for k in range(predict_iters):
                X, Y = next(data)
                X, Y = X.to(device), Y.to(device)
                loss, _ = self.model(X, Y)
                losses[k] = loss.item()
            output[split] = losses.mean().item()
        self.model.train()
        return output


    def save_model(self, output_dir: str) -> None:
        '''Save the trained model, then you can reload it with from_pretrained()'''
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(output_dir, 'model.pt'))

    def train(self):
        if resume_from_checkpoint:
            model_path = os.path.join(output_dir, "model.pt")
            self.model.from_pretrained(model_path)

        self.model.to(device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

        dataloader = self.get_dataloader(self.train_dataset)
        data = iter(dataloader)

        for iteration in range(batch_iterations):
            if iteration % eval_interval == 0 or iteration == batch_iterations - 1:
                predicted_loss = self.predict_loss()
                print(
                    f"STEP {iteration} --> Training loss: {predicted_loss['train']:.4f} || Validation loss: {predicted_loss['val']:.4f}"
                )

            xb, yb = next(data)
            xb, yb = xb.to(device), yb.to(device)
            loss, _ = self.model(xb, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if save_checkpoint:
            with torch.no_grad():
                self.save_model(output_dir)


def main():
    args = ModelArgs()
    model = GPT(args)
    trainer = Trainer(model)

    print(f'Device: {device}')
    # TODO: add model parameters
    
    trainer.train()

if __name__ == "__main__":    
    main()