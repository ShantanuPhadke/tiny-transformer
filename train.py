from data_processing.FileProcessor import FileProcessor
from data_processing.Vocabulary import VocabularyManager
from data_processing.data_management import TrainValidationSplitter, Chunker

from tokenization.CharacterTokenizer import CharacterTokenizer

from models.language_models import BigramLanguageModel

import torch

# Manage Hyperparameters
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
batchsize = 32
blocksize = 8
learning_rate = 1e-3
train_size = 0.9
max_iterations = 3000
evaluation_interval = 300
eval_iters = 200


url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

shakespeare_processor = FileProcessor(url)
shakespeare_processor.save_file("shakespeare.txt", shakespeare_processor.fetch_file())
shakespeare_content = shakespeare_processor.get_file_content("shakespeare.txt")

vocab_manager = VocabularyManager()
vocab_manager.generate_vocabulary(shakespeare_content)

shakespeare_character_tokenizer = CharacterTokenizer(vocab_manager)
shakespeare_character_encoding = shakespeare_character_tokenizer.encode(shakespeare_content)

train_val_splitter = TrainValidationSplitter(shakespeare_character_encoding, train_size=train_size)
train_data, validation_data = train_val_splitter.split()

bigram_lm = BigramLanguageModel(vocab_manager.get_vocabulary_size())
bigram_lm.to(device)  # Move the model to the appropriate device

train_data_chunker = Chunker(train_data, batchsize=batchsize, blocksize=blocksize)
validation_data_chunker = Chunker(validation_data, batchsize=batchsize, blocksize=blocksize)

# Function to estimate the loss across multiple batches
@torch.no_grad() # No need to track gradients for evaluation (no backward pass)
def estimate_loss(model, data, eval_iters):
    out = {}
    model.eval()  # Set the model to evaluation mode
    for split in ['train', 'validation']:
        if split == 'train':
            batches = train_data_chunker
        else:
            batches = validation_data_chunker
        losses = torch.zeros(eval_iters)
        for l in range(eval_iters):
            xb, yb = batches.get_batch()  # Get a batch of data
            logits, loss = model(xb, yb)  # Forward pass
            losses[l] = loss.item()
        out[split] = losses.mean()
    model.train()  # Set the model back to training mode
    return out

# Training Loop with an Optimizer
optimizer = torch.optim.AdamW(bigram_lm.parameters(), lr=learning_rate)
for steps in range(max_iterations):
    # Every once in a while evaluate the loss on training and validation sets
    if steps % evaluation_interval == 0:
        losses = estimate_loss(bigram_lm, train_data_chunker, eval_iters)
        print(f"Step {steps}: Train Loss: {losses['train']:.4f}, Validation Loss: {losses['validation']:.4f}")
    # Get a random chunk / batch of data
    xb, yb = train_data_chunker.get_batch()
    # Generate predictions and loss for the batch
    logits, loss = bigram_lm(xb, yb)
    optimizer.zero_grad(set_to_none=True)  # Reset gradients to zero before backpropagation
    # Backpropagation and optimization would go here (not implemented in this example)
    loss.backward()  # Backpropagate the loss - Calculate the difference in current predictions and actual values
    optimizer.step() # make a step with the optimizer
print(loss.item())  # Print the loss after the last step

print("Training complete.")
context = torch.zeros((1, 1), dtype=torch.long, device=device)  # Start with a single character (e.g., the first character in the vocabulary)
generated_sequence_raw = bigram_lm.generate(context, max_new_tokens=400) # Generate 10 new characters where character n is based on character n-1
generated_sequence_decoding = shakespeare_character_tokenizer.decode(torch.tensor(generated_sequence_raw[0].tolist()))
print("Generated sequence after training:", generated_sequence_raw)  # Print the generated sequence as indices
print()
print("Decoded generated sequence after training:", generated_sequence_decoding)





