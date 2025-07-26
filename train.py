from data_processing.FileProcessor import FileProcessor
from data_processing.Vocabulary import VocabularyManager
from data_processing.data_management import TrainValidationSplitter, Chunker

from tokenization.CharacterTokenizer import CharacterTokenizer

from models.language_models import BigramLanguageModel

import torch

# Use Apple Silicon if available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

shakespeare_processor = FileProcessor(url)
shakespeare_processor.save_file("shakespeare.txt", shakespeare_processor.fetch_file())
shakespeare_content = shakespeare_processor.get_file_content("shakespeare.txt")

vocab_manager = VocabularyManager()
vocab_manager.generate_vocabulary(shakespeare_content)

shakespeare_character_tokenizer = CharacterTokenizer(vocab_manager)
shakespeare_character_encoding = shakespeare_character_tokenizer.encode(shakespeare_content)

train_val_splitter = TrainValidationSplitter(shakespeare_character_encoding, train_size=0.9)
train_data, validation_data = train_val_splitter.split()

train_data_chunker = Chunker(train_data, batchsize=4, blocksize=8)

bigram_lm = BigramLanguageModel(vocab_manager.get_vocabulary_size())

# Training Loop with an Optimizer
batch_size = 32
optimizer = torch.optim.AdamW(bigram_lm.parameters(), lr=1e-3)
train_data_chunker = Chunker(train_data, batchsize=batch_size, blocksize=8)
for steps in range(10000):
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
idx = torch.zeros((1, 1), dtype=torch.long)  # Start with a single character (e.g., the first character in the vocabulary)
generated_sequence_raw = bigram_lm.generate(idx, max_new_tokens=400) # Generate 10 new characters where character n is based on character n-1
generated_sequence_decoding = shakespeare_character_tokenizer.decode(torch.tensor(generated_sequence_raw[0].tolist()))
print("Generated sequence after training:", generated_sequence_raw)  # Print the generated sequence as indices
print()
print("Decoded generated sequence after training:", generated_sequence_decoding)





