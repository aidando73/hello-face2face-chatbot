from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn

class CustomQwenModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        # Add custom layers
        self.custom_layer1 = nn.Linear(base_model.config.hidden_size, base_model.config.hidden_size)
        self.custom_layer2 = nn.Linear(base_model.config.hidden_size, base_model.config.hidden_size)
        self.activation = nn.GELU()
        
    def forward(self, input_ids, attention_mask=None, **kwargs):
        # Get base model outputs
        outputs = self.base_model(input_ids, attention_mask=attention_mask, **kwargs)
        
        # Get the last hidden states
        hidden_states = outputs.last_hidden_state
        
        # Apply custom layers
        custom_output = self.custom_layer1(hidden_states)
        custom_output = self.activation(custom_output)
        custom_output = self.custom_layer2(custom_output)
        
        # Update the outputs with modified hidden states
        outputs.last_hidden_state = custom_output
        
        return outputs

def load_and_modify_qwen():
    # Load the base model and tokenizer
    model_name = "Qwen/Qwen2.5-7B"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    # Create custom model
    custom_model = CustomQwenModel(base_model)
    custom_model = custom_model.to(base_model.device)
    
    return custom_model, tokenizer

if __name__ == "__main__":
    model, tokenizer = load_and_modify_qwen()
    print("Custom model loaded successfully!")
    
    # Example usage
    prompt = "Hello, how are you?"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.base_model.device)
    
    # Generate response
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Response: {response}") 