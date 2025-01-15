# model_utils.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from safetensors import safe_open

# Custom model class
class BertForMultiOutputRegression(nn.Module):
    def __init__(self):
        super(BertForMultiOutputRegression, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.regressor = nn.Linear(self.bert.config.hidden_size, 3)  # 3 outputs: content, grammar, structure

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.regressor(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(logits, labels)

        return {'loss': loss, 'logits': logits} if loss is not None else {'logits': logits}


# Load model and tokenizer
def load_model(checkpoint_dir='./model'):
    tokenizer = BertTokenizer.from_pretrained(checkpoint_dir)
    model = BertForMultiOutputRegression()
    
    # Load model weights
    state_dict = {}
    with safe_open(f"{checkpoint_dir}/model.safetensors", framework="pt", device="cpu") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)
    model.load_state_dict(state_dict)
    model.eval()

    return model, tokenizer


# Prediction function
def predict(input_text, model, tokenizer, max_length=512):
    inputs = tokenizer(
        input_text,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs['logits'].squeeze()
        
        content_score = F.sigmoid(logits[0]) * 3  # Scale to 0-3
        grammar_score = F.sigmoid(logits[1]) * 5  # Scale to 0-5
        structure_score = F.sigmoid(logits[2]) * 5  # Scale to 0-5
    
    return {
        'content_relevancy_score': content_score.item(),
        'grammar_score': grammar_score.item(),
        'structure_score': structure_score.item()
    }
