from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_qwen_model():
    # Load the model and tokenizer
    model_name = "Qwen/Qwen2.5-7B"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # device_map="auto",  # This will automatically handle device placement
        torch_dtype=torch.float16,  # Use float16 for memory efficiency
        trust_remote_code=True
    )
    
    return model, tokenizer

if __name__ == "__main__":
    model, tokenizer = load_qwen_model()
    print("Model and tokenizer loaded successfully!")

    # Print the type of model and tokenizer
    print(f"Model type: {type(model)}")
    print(f"Tokenizer type: {type(tokenizer)}")
    print(isinstance(model, torch.nn.Module))  # Will print True
    print(isinstance(tokenizer, torch.nn.Module))  # Will print False
    
    # Example usage
    prompt = "Hello, how are you?"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate response
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Response: {response}")

    # Get all weights as a PyTorch state dict
    state_dict = model.state_dict()

    # Each key in the state dict is a PyTorch tensor
    for name, param in state_dict.items():
        print(f"Layer: {name}, Shape: {param.shape}") 