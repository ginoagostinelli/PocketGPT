def sample_sequence(model, tokenizer, prompt='', steps=100):
    if prompt == '':
        prompt = '<|endoftext|>'
    encoded_input = tokenizer(prompt, return_tensors='pt') #.to(device)
    x = encoded_input['input_ids']
    y = model.generate(x, max_new_tokens=steps)
    return tokenizer.decode(y.cpu().squeeze())
