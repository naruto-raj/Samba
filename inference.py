import torch
from torch import nn
from pathlib import Path
from transformers import AutoTokenizer
from lit_gpt.model import GPT, Config

def load_model(checkpoint_path, device):
    config = Config.from_name("Samba_270M")
    model = GPT(config)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model

def generate(model, tokenizer, prompt, device, max_new_tokens=1024, temperature=0.8, top_k=200):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated_tokens = input_ids[0].tolist()

    # Monkey-patch the forward method of the model to ensure dtype consistency
    original_forward = model.forward
    
    def new_forward(idx):
        # Ensure the input is long
        idx = idx.long()
        
        # Generate rope cache if it doesn't exist
        if model.rope_cache is None:
            model.rope_cache = model.build_rope_cache(idx, model.max_len)
        
        # Get the rope cache and ensure it's in the correct dtype
        rope = model.rope_cache
        model_dtype = next(model.parameters()).dtype
        rope = tuple(r.to(dtype=model_dtype) for r in rope)
        
        x = model.transformer.wte(idx)
        x = x.to(model_dtype)
        
        # Ensure all model parameters are in the same dtype
        for param in model.parameters():
            param.data = param.data.to(model_dtype)
        
        max_seq_length = model.config.block_size
        for i, block in enumerate(model.transformer.h):
            x, _ = block(x, rope, max_seq_length)
        
        x = model.transformer.ln_f(x)
        return model.lm_head(x)
    
    model.forward = new_forward
    
    for _ in range(max_new_tokens):
        idx_cond = torch.tensor(generated_tokens, dtype=torch.long, device=device).unsqueeze(0)
        
        with torch.no_grad():
            logits = model(idx_cond)
        
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        probs = torch.nn.functional.softmax(logits, dim=-1)
        
        next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
        
        generated_tokens.append(next_token.item())


        if next_token.item() == tokenizer.eos_token_id or next_token.item() == 128009:
            break

    model.forward = original_forward
    
    return tokenizer.decode(generated_tokens)

def main():
    checkpoint_path = Path("iter-1003200-ckpt.pth")
    tokenizer_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(checkpoint_path, device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    prompt = "One upon a time "
    output = generate(model, tokenizer, prompt, device)
    print(output)

if __name__ == "__main__":
    main()
