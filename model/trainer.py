from gpt import GPT
import torch
import os
import numpy as np

# hyperparameters
batch_size = 32 
batch_iterations = 2500
eval_interval = 250
learning_rate = 3e-4
predict_iters = 220
context_size = 12
# ------------

def get_batch(split):
    data = train_dataset if split == 'train' else val_dataset
    ix = torch.randint(len(data) - context_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+context_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+context_size]).astype(np.int64)) for i in ix])
    return x, y


@torch.no_grad()
def predict_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(predict_iters)
        for k in range(predict_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

if __name__ == "__main__":
    
    model = GPT()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # TODO: add a data loader 
    train_dataset = np.memmap(os.path.join('', 'train.bin'), dtype=np.uint16, mode='r')
    val_dataset = np.memmap(os.path.join('', 'val.bin'), dtype=np.uint16, mode='r')

    for iter in range(batch_iterations):

        if iter % eval_interval == 0 or iter == batch_iterations - 1:
            predicted_loss = predict_loss()
            print(f"STEP {iter} --> Training loss: {predicted_loss['train']:.4f} || Validation loss: {predicted_loss['val']:.4f}")

        xb, yb = get_batch("train")

        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        
        loss.backward()
        optimizer.step()