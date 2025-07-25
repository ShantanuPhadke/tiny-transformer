from data_processing.FileProcessor import FileProcessor
from data_processing.Vocabulary import VocabularyManager

from tokenization.CharacterTokenizer import CharacterTokenizer

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

shakespeare_processor = FileProcessor(url)
shakespeare_processor.save_file("shakespeare.txt", shakespeare_processor.fetch_file())
shakespeare_content = shakespeare_processor.get_file_content("shakespeare.txt")

vocab_manager = VocabularyManager()
vocab_manager.generate_vocabulary(shakespeare_content)
print(vocab_manager)

shakespeare_character_tokenizer = CharacterTokenizer(vocab_manager)
shakespeare_character_sample_encoding = shakespeare_character_tokenizer.encode(shakespeare_content[:100])


shakespeare_character_sample_decoding = shakespeare_character_tokenizer.decode(shakespeare_character_sample_encoding)


