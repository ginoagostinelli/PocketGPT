from gpt import GPT
import os
import torch
import numpy as np

# config
batch_size = 32
batch_iterations = 2500
eval_interval = 220
learning_rate = 3e-4
predict_iters = 220
context_size = 12
save_checkpoint = True
# save_steps = 1
resume_from_checkpoint = True
# ------------
output_dir = os.path.join(os.path.dirname(__file__), 'output')


def get_batch(split):
    data = train_dataset if split == 'train' else val_dataset
    ix = torch.randint(len(data) - context_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+context_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+context_size]).astype(np.int64)) for i in ix])
    return x, y


@torch.no_grad()
def predict_loss():
    output = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(predict_iters)
        for k in range(predict_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        output[split] = losses.mean()
    model.train()
    return output


def save_model(output_dir: str) -> None:
    '''Save the trained model, then you can reload it with from_pretrained()'''
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, 'model.pt'))


def from_pretrained(output_dir: str) -> None:
    ''' Load the trained model '''
    if not os.path.exists(os.path.join(output_dir, 'model.pt')):
        raise FileNotFoundError(f'"model.pt" not found in "{output_dir}"')
    
    model.load_state_dict(torch.load(os.path.join(output_dir, 'model.pt')))


if __name__ == "__main__":    
    model = GPT()
    if resume_from_checkpoint:
        from_pretrained(output_dir)
    
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
    
    if save_checkpoint:
        save_model(output_dir)