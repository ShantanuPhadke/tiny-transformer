class VocabularyManager:
    def __init__(self):
        self.chars = set()

    def generate_vocabulary(self, text):
        self.chars = sorted(list(set(text)))

    def get_vocabulary_size(self):
        return len(self.chars)

    def get_vocabulary(self):
        return self.chars

    def __str__(self):
        return f"Vocabulary: {''.join(self.chars)}, Size: {self.get_vocabulary_size()}"
