from data_processing.FileProcessor import FileProcessor
from data_processing.Vocabulary import VocabularyManager
from data_processing.data_management import TrainValidationSplitter, Chunker

from tokenization.CharacterTokenizer import CharacterTokenizer

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

shakespeare_processor = FileProcessor(url)
shakespeare_processor.save_file("shakespeare.txt", shakespeare_processor.fetch_file())
shakespeare_content = shakespeare_processor.get_file_content("shakespeare.txt")

vocab_manager = VocabularyManager()
vocab_manager.generate_vocabulary(shakespeare_content)
print(vocab_manager)

shakespeare_character_tokenizer = CharacterTokenizer(vocab_manager)
shakespeare_character_encoding = shakespeare_character_tokenizer.encode(shakespeare_content)

train_val_splitter = TrainValidationSplitter(shakespeare_character_encoding, train_size=0.9)
train_data, validation_data = train_val_splitter.split()

data_chunker = Chunker(train_data, batchsize=4, blocksize=8)
random_batch = data_chunker.get_batch()
data_chunker.print_batch_info(*random_batch)





