from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn

class CustomQwenModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        
        # Get the embedding layer from the base model
        self.embedding = base_model.get_input_embeddings()
        
        # Add custom input processing layers
        self.input_projection = nn.Linear(
            base_model.config.hidden_size,
            base_model.config.hidden_size
        )
        self.layer_norm = nn.LayerNorm(base_model.config.hidden_size)
        
    def forward(self, input_ids, attention_mask=None, **kwargs):
        # Get the input embeddings
        inputs_embeds = self.embedding(input_ids)
        
        # Apply custom input processing
        processed_embeds = self.input_projection(inputs_embeds)
        processed_embeds = self.layer_norm(processed_embeds)
        
        # Pass the processed embeddings to the base model
        # Note: We use inputs_embeds instead of input_ids to bypass the embedding layer
        outputs = self.base_model(
            inputs_embeds=processed_embeds,
            attention_mask=attention_mask,
            **kwargs
        )
        
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