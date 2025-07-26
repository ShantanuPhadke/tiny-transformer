import torch

class TrainValidationSplitter:
    def __init__(self, data, train_size=0.8):
        self.data = data
        self.train_size = train_size
    
    def split(self):
        train_length = int(len(self.data) * self.train_size)
        train_data = self.data[:train_length]
        validation_data = self.data[train_length:]
        return train_data, validation_data

    def __str__(self):
        return f"Train size: {len(self.data) * self.train_size}, Validation size: {len(self.data) * (1 - self.train_size)}"

class Chunker:
    def __init__(self, data, batchsize, blocksize):
        self.data = data
        self.batchsize = batchsize # Number of sequences processed in parallel
        self.blocksize = blocksize # Length of each block

    def get_batch(self):
        # self.batchsize number of random starting points for the sequences
        randx = torch.randint(len(self.data) - self.blocksize, (self.batchsize,))  # Randomly select starting points for each sequence
        # x values of each batch
        batchx = torch.stack([self.data[i:i + self.blocksize] for i in randx])
        # y values of each batch (next character in the sequence)
        batchy = torch.stack([self.data[i + 1:i + self.blocksize + 1] for i in randx])
        return batchx, batchy

    def print_batch_info(self, batchx, batchy):
        for i in range(self.batchsize):
            print("Element ", i+1, " of the batch:")
            for j in range(self.blocksize):
                batch_input = batchx[i, :j+1]
                batch_output = batchy[i, j]
                print("When the batch input is:", batch_input, "the output is:", batch_output)
            print("--------------------------------------------------------------")
            print()
