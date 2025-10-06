# Ví Dụ Sử Dụng Mamba - Code Examples

## Mục Lục
1. [Basic Usage](#1-basic-usage)
2. [Advanced Configuration](#2-advanced-configuration)
3. [Training Examples](#3-training-examples)
4. [Inference Examples](#4-inference-examples)
5. [Custom Applications](#5-custom-applications)

---

## 1. Basic Usage

### 1.1 Sử Dụng Mamba Block Cơ Bản

```python
import torch
from mamba_ssm import Mamba

# Khởi tạo model
model = Mamba(
    d_model=512,        # Dimension của model
    d_state=16,         # SSM state expansion factor
    d_conv=4,           # Kernel size của convolution
    expand=2,           # Expansion factor (d_inner = expand * d_model)
).to("cuda")

# Forward pass
batch_size = 2
seq_length = 128
x = torch.randn(batch_size, seq_length, 512).to("cuda")
output = model(x)

print(f"Input shape: {x.shape}")    # [2, 128, 512]
print(f"Output shape: {output.shape}")  # [2, 128, 512]
```

### 1.2 Sử Dụng Mamba-2

```python
from mamba_ssm import Mamba2

# Mamba-2 có cải tiến với multi-head và chunk processing
model = Mamba2(
    d_model=768,
    d_state=64,         # Thường lớn hơn Mamba v1
    d_conv=4,
    expand=2,
    headdim=128,        # Dimension per head
    chunk_size=256,     # Chunk size cho processing
).to("cuda")

x = torch.randn(2, 256, 768).to("cuda")
output = model(x)

print(f"Output shape: {output.shape}")  # [2, 256, 768]
```

### 1.3 Load Pre-trained Model

```python
from mamba_ssm import MambaLMHeadModel
from transformers import AutoTokenizer

# Load pre-trained model
model = MambaLMHeadModel.from_pretrained(
    "state-spaces/mamba-130m"
).to("cuda")
model.eval()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "EleutherAI/gpt-neox-20b"
)

# Test generation
prompt = "The future of AI is"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

with torch.no_grad():
    output_ids = model.generate(
        input_ids,
        max_length=50,
        temperature=0.7,
        top_p=0.9,
    )

generated_text = tokenizer.decode(output_ids[0])
print(generated_text)
```

---

## 2. Advanced Configuration

### 2.1 Custom Model Configuration

```python
from mamba_ssm import Mamba

# Configuration với nhiều tùy chọn
model = Mamba(
    d_model=1024,
    d_state=32,           # Tăng state dimension
    d_conv=8,             # Larger convolution kernel
    expand=3,             # Larger expansion
    dt_rank="auto",       # Auto-calculate dt_rank
    dt_min=0.001,         # Min timestep
    dt_max=0.1,           # Max timestep
    dt_init="random",     # Initialization strategy
    dt_scale=1.0,         # Scale factor
    dt_init_floor=1e-4,   # Floor for dt initialization
    conv_bias=True,       # Use bias in convolution
    bias=False,           # No bias in linear layers
    use_fast_path=True,   # Use optimized kernels
).to("cuda")
```

### 2.2 Mamba-2 với RMSNorm

```python
from mamba_ssm import Mamba2

model = Mamba2(
    d_model=1024,
    d_state=128,
    d_conv=4,
    expand=2,
    headdim=64,
    ngroups=4,            # Number of groups
    A_init_range=(1, 16), # Range for A initialization
    dt_min=0.001,
    dt_max=0.1,
    dt_init_floor=1e-4,
    dt_limit=(0.0, float("inf")),  # Timestep limits
    rmsnorm=True,         # Use RMSNorm instead of LayerNorm
    norm_before_gate=True,  # RMSNorm before gating
    chunk_size=512,       # Larger chunks
    use_mem_eff_path=True,  # Memory-efficient path
).to("cuda")
```

### 2.3 Building Complete Language Model

```python
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig

# Create custom config
config = MambaConfig(
    d_model=1024,
    n_layer=48,              # Number of Mamba blocks
    vocab_size=50280,        # Vocabulary size
    ssm_cfg={
        "d_state": 16,
        "d_conv": 4,
        "expand": 2,
    },
    rms_norm=True,
    fused_add_norm=True,     # Fused operations
    residual_in_fp32=True,   # Keep residuals in fp32
)

# Create model from config
model = MambaLMHeadModel(config).to("cuda")

print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
```

---

## 3. Training Examples

### 3.1 Simple Training Loop

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from mamba_ssm import Mamba

# Setup model
model = Mamba(d_model=512, d_state=16).to("cuda")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# Training loop
model.train()
for epoch in range(num_epochs):
    for batch in dataloader:
        # batch: (B, L, D)
        inputs = batch['input'].to("cuda")
        targets = batch['target'].to("cuda")
        
        # Forward pass
        outputs = model(inputs)
        
        # Reshape for loss computation
        outputs = outputs.reshape(-1, outputs.size(-1))
        targets = targets.reshape(-1)
        
        # Compute loss
        loss = criterion(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Update weights
        optimizer.step()
        
        if step % 100 == 0:
            print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")
```

### 3.2 Training với Mixed Precision (AMP)

```python
from torch.cuda.amp import autocast, GradScaler

model = Mamba(d_model=768).to("cuda")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scaler = GradScaler()

model.train()
for batch in dataloader:
    inputs = batch['input'].to("cuda")
    targets = batch['target'].to("cuda")
    
    optimizer.zero_grad()
    
    # Forward pass với autocast
    with autocast(dtype=torch.float16):
        outputs = model(inputs)
        loss = criterion(outputs, targets)
    
    # Backward pass với gradient scaling
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
```

### 3.3 Training Language Model

```python
from mamba_ssm import MambaLMHeadModel
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# Model và tokenizer
model = MambaLMHeadModel.from_pretrained("state-spaces/mamba-130m").to("cuda")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

# Optimizer và scheduler
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=6e-4,
    betas=(0.9, 0.95),
    weight_decay=0.1,
)

num_training_steps = len(dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_training_steps // 10,
    num_training_steps=num_training_steps,
)

# Training loop
model.train()
for epoch in range(num_epochs):
    for batch_idx, batch in enumerate(dataloader):
        # Tokenize
        tokens = tokenizer(
            batch['text'],
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=2048,
        ).to("cuda")
        
        # Forward pass
        outputs = model(
            input_ids=tokens['input_ids'],
            labels=tokens['input_ids'],  # For language modeling
        )
        
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Update
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        # Logging
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
```

---

## 4. Inference Examples

### 4.1 Text Generation Cơ Bản

```python
from mamba_ssm import MambaLMHeadModel
from transformers import AutoTokenizer

model = MambaLMHeadModel.from_pretrained("state-spaces/mamba-370m").to("cuda")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
model.eval()

def generate_text(prompt, max_length=100, temperature=0.8, top_p=0.95):
    """Generate text from prompt"""
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to("cuda")
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
        )
    
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text

# Test generation
prompts = [
    "Once upon a time",
    "The meaning of life is",
    "In the year 2050,",
]

for prompt in prompts:
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {generate_text(prompt)}")
```

### 4.2 Batch Generation

```python
def batch_generate(prompts, model, tokenizer, max_length=100):
    """Generate text for multiple prompts efficiently"""
    # Tokenize all prompts
    inputs = tokenizer(
        prompts,
        return_tensors='pt',
        padding=True,
        truncation=True,
    ).to("cuda")
    
    with torch.no_grad():
        output_ids = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=max_length,
            temperature=0.8,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode all outputs
    generated_texts = [
        tokenizer.decode(ids, skip_special_tokens=True)
        for ids in output_ids
    ]
    
    return generated_texts

# Test batch generation
prompts = [
    "Artificial intelligence will",
    "Climate change is",
    "Space exploration allows us to",
]

results = batch_generate(prompts, model, tokenizer)
for prompt, result in zip(prompts, results):
    print(f"\nPrompt: {prompt}")
    print(f"Result: {result}")
```

### 4.3 Interactive Generation

```python
def interactive_chat():
    """Interactive chat với Mamba model"""
    model = MambaLMHeadModel.from_pretrained("state-spaces/mamba-790m").to("cuda")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    model.eval()
    
    print("Mamba Chat (type 'quit' to exit)")
    conversation_history = ""
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'quit':
            break
        
        # Add to conversation history
        conversation_history += f"Human: {user_input}\nAssistant:"
        
        # Generate response
        input_ids = tokenizer(conversation_history, return_tensors='pt').input_ids.to("cuda")
        
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=100,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                eos_token_id=tokenizer.encode('\n')[0],  # Stop at newline
            )
        
        response = tokenizer.decode(
            output_ids[0][len(input_ids[0]):],
            skip_special_tokens=True
        )
        
        print(f"Assistant: {response}")
        conversation_history += f" {response}\n"

# Run interactive chat
interactive_chat()
```

### 4.4 Streaming Generation

```python
def stream_generate(prompt, model, tokenizer, max_new_tokens=100):
    """Generate text token by token (streaming)"""
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to("cuda")
    
    print(f"Prompt: {prompt}")
    print("Generated: ", end="", flush=True)
    
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Forward pass
            outputs = model(input_ids)
            logits = outputs.logits
            
            # Get next token
            next_token_logits = logits[:, -1, :] / 0.8  # temperature
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            # Decode and print
            token_text = tokenizer.decode(next_token[0])
            print(token_text, end="", flush=True)
            
            # Stop at end of sentence
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    print()  # Newline

# Test streaming
stream_generate("The key to success is", model, tokenizer)
```

---

## 5. Custom Applications

### 5.1 Sentiment Analysis với Mamba

```python
import torch.nn as nn
from mamba_ssm import Mamba

class MambaSentimentClassifier(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_classes=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.mamba = Mamba(d_model=d_model, d_state=16)
        self.classifier = nn.Linear(d_model, n_classes)
    
    def forward(self, input_ids):
        # Embed
        x = self.embedding(input_ids)  # (B, L, D)
        
        # Mamba processing
        x = self.mamba(x)  # (B, L, D)
        
        # Pool (mean over sequence)
        x = x.mean(dim=1)  # (B, D)
        
        # Classify
        logits = self.classifier(x)  # (B, n_classes)
        return logits

# Usage
model = MambaSentimentClassifier(vocab_size=50000).to("cuda")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# Training
model.train()
for batch in dataloader:
    input_ids = batch['input_ids'].to("cuda")
    labels = batch['labels'].to("cuda")
    
    logits = model(input_ids)
    loss = criterion(logits, labels)
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### 5.2 Sequence-to-Sequence với Mamba

```python
class MambaSeq2Seq(nn.Module):
    def __init__(self, vocab_size, d_model=512):
        super().__init__()
        self.encoder_embed = nn.Embedding(vocab_size, d_model)
        self.decoder_embed = nn.Embedding(vocab_size, d_model)
        
        # Encoder
        self.encoder = nn.ModuleList([
            Mamba(d_model=d_model, d_state=16)
            for _ in range(6)
        ])
        
        # Decoder
        self.decoder = nn.ModuleList([
            Mamba(d_model=d_model, d_state=16)
            for _ in range(6)
        ])
        
        self.output_proj = nn.Linear(d_model, vocab_size)
    
    def encode(self, src):
        x = self.encoder_embed(src)
        for layer in self.encoder:
            x = x + layer(x)  # Residual
        return x
    
    def decode(self, tgt, encoder_output):
        x = self.decoder_embed(tgt)
        
        for layer in self.decoder:
            x = x + layer(x)
            # Cross-attention alternative: add encoder output
            x = x + encoder_output[:, :x.size(1), :]
        
        return self.output_proj(x)
    
    def forward(self, src, tgt):
        encoder_output = self.encode(src)
        decoder_output = self.decode(tgt, encoder_output)
        return decoder_output

# Usage
model = MambaSeq2Seq(vocab_size=50000).to("cuda")
```

### 5.3 Time Series Forecasting

```python
class MambaTimeSeriesForecaster(nn.Module):
    def __init__(self, input_size, d_model=256, forecast_horizon=10):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        
        self.mamba_layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=16)
            for _ in range(4)
        ])
        
        self.output_proj = nn.Linear(d_model, forecast_horizon * input_size)
        self.forecast_horizon = forecast_horizon
        self.input_size = input_size
    
    def forward(self, x):
        # x: (B, T, input_size)
        x = self.input_proj(x)  # (B, T, d_model)
        
        # Process with Mamba
        for layer in self.mamba_layers:
            x = x + layer(x)
        
        # Use last timestep for forecasting
        x = x[:, -1, :]  # (B, d_model)
        
        # Project to forecast
        forecast = self.output_proj(x)  # (B, forecast_horizon * input_size)
        forecast = forecast.reshape(-1, self.forecast_horizon, self.input_size)
        
        return forecast

# Usage
model = MambaTimeSeriesForecaster(input_size=10, forecast_horizon=20).to("cuda")

# Training
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

for batch in dataloader:
    history = batch['history'].to("cuda")  # (B, T_history, features)
    future = batch['future'].to("cuda")    # (B, T_future, features)
    
    forecast = model(history)
    loss = criterion(forecast, future)
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### 5.4 Document Embedding

```python
class MambaDocumentEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=768, embedding_dim=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        self.mamba_layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=16)
            for _ in range(8)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.pooler = nn.Linear(d_model, embedding_dim)
    
    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        
        # Process with Mamba
        for layer in self.mamba_layers:
            x = x + layer(x)
        
        x = self.norm(x)
        
        # Mean pooling (considering attention mask)
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand(x.size())
            sum_embeddings = torch.sum(x * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            x = sum_embeddings / sum_mask
        else:
            x = x.mean(dim=1)
        
        # Project to embedding space
        embedding = self.pooler(x)
        
        # Normalize
        embedding = nn.functional.normalize(embedding, p=2, dim=1)
        
        return embedding

# Usage for semantic search
model = MambaDocumentEncoder(vocab_size=50000).to("cuda")
model.eval()

# Encode documents
with torch.no_grad():
    doc_embeddings = model(doc_ids)  # (N_docs, embedding_dim)
    query_embedding = model(query_ids)  # (1, embedding_dim)
    
    # Compute similarity
    similarities = torch.mm(query_embedding, doc_embeddings.t())
    top_k = similarities.topk(5)
    
    print(f"Top 5 similar documents: {top_k.indices}")
```

---

## Best Practices

### 1. Memory Management
```python
# Clear cache regularly
torch.cuda.empty_cache()

# Use gradient checkpointing for large models
from torch.utils.checkpoint import checkpoint

def forward_with_checkpointing(x):
    for layer in self.layers:
        x = checkpoint(layer, x)
    return x
```

### 2. Initialization
```python
# Proper initialization
def init_weights(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)

model.apply(init_weights)
```

### 3. Learning Rate Scheduling
```python
from torch.optim.lr_scheduler import CosineAnnealingLR

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

for epoch in range(num_epochs):
    # Training...
    scheduler.step()
```

### 4. Checkpointing
```python
# Save checkpoint
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}
torch.save(checkpoint, 'checkpoint.pth')

# Load checkpoint
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

---

## Tài Liệu Tham Khảo

- Official Repository: https://github.com/state-spaces/mamba
- Hugging Face Models: https://huggingface.co/state-spaces
- Papers: 
  - Mamba: https://arxiv.org/abs/2312.00752
  - Mamba-2: https://arxiv.org/abs/2405.21060
