
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
class TextEncoder(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", pretrained=True, trainable=False):
        super().__init__()
        if pretrained:
            self.model = DistilBertModel.from_pretrained(model_name)
        else:
            self.model = DistilBertModel(config=DistilBertConfig())
        for p in self.model.parameters():
            p.requires_grad = trainable
        self.target_token_idx = 0
    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]
class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=1024,
        dropout=0.1
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

class CKQP_Model(nn.Module):
    def __init__(
        self,   
        temperature=1.,
        image_embedding=768,
        text_embedding=768,
        max_length=256
    ):
        super().__init__()
        self.text_encoder = TextEncoder()
        self.question_encoder = TextEncoder()

        self.question_projection = ProjectionHead(embedding_dim=text_embedding)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)
        self.temperature = temperature
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.max_length = max_length
    def tokenize(self, texts):
        return self.tokenizer(
            list([str(text) for text in texts]), padding=True, truncation=True, max_length=self.max_length
        )
    def forward(self, text_features, question_features, mask1, mask2):
        # mask= torch.ones(16, 1, 1, 20).to(device)
        text_features = self.text_encoder(text_features,mask1)
        question_features = self.question_encoder(question_features,mask2)
        question_embeddings = self.question_projection(question_features)
        text_embeddings = self.text_projection(text_features)

        logits = (text_embeddings @ question_embeddings.T) / self.temperature
        questions_similarity = question_features @ question_features.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (questions_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        texts_loss = cross_entropy(logits, targets, reduction='none')
        questions_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (questions_loss + texts_loss) / 2.0 # shape: (batch_size)
        return loss.mean()
    def score(self,questions,answers):
        text_features = self.text_encoder(torch.tensor(answers['input_ids'], device=device),torch.tensor(answers['attention_mask'],device=device))
        question_features = self.question_encoder(torch.tensor(questions['input_ids'],device=device),torch.tensor(questions['attention_mask'],device=device))
        question_embeddings = self.question_projection(question_features)
        text_embeddings = self.text_projection(text_features)
        return question_embeddings, text_embeddings
    def best_answer(self, question, answers):
        q_tok = self.tokenize([question])
        a_tok = self.tokenize(answers)
        scores = self.score(q_tok,a_tok)
        scores = [torch.nn.functional.cosine_similarity(scores[0], scores[1][i]).item() for i in range(len(scores[1]))]
        ind = scores.index(max(scores))
        return (answers[ind], ind, max(scores))
    def best_question(self, questions, answer):
        q_tok = self.tokenize(questions)
        a_tok = self.tokenize([answer])
        scores = self.score(q_tok,a_tok)
        scores = [torch.nn.functional.cosine_similarity(scores[0][i], scores[1]).item() for i in range(len(scores[0]))]
        ind = scores.index(max(scores))
        return (questions[ind], ind, max(scores))
def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

def load_model(path="SQuAD_CQKP.pt",device='cpu'):
    import os
    files = [f for f in os.listdir('.') if os.path.isfile(f)]
    if "SQuAD_CQKP.pt" not in files:
        wandb.restore('SQuAD_CQKP.pt', run_path="boopysaur/CQKP/1nfqx9u0")
    model = CKQP_Model().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.eval()