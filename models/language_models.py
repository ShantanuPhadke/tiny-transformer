import torch
import torch.nn as nn
import torch.nn.functional as F

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, vocab_size)
    
    # Sidenote: y = None by default because during generation we won't pass in y values
    def forward(self, x, y=None):
        logits = self.embedding(x) # (Batch Size, Block Size, Vocab Size)
        if y is None:
            return logits, None
        else:
            #print("Logits shape:", str(logits.shape) + ", Logits.view(-1) Shape: " + str(logits.view(-1, self.vocab_size).shape))
            #print("Y shape:", str(y.shape) +  ", Y.view(-1) Shape: " + str(y.view(-1).shape))
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), y.view(-1))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx = (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # Getting the probabilities for the last character in the sequence
            logits = logits[:, -1, :] # Size = (Batch Size, 1, Vocab Size)
            # Converting logits to probabilities
            probs = F.softmax(logits, dim=-1) # Size = (Batch Size, 1, Vocab Size)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1) # Append the new character to the sequence
        # Returns the newly generated sequence
        return idx


