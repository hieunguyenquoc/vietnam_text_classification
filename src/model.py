import torch
import torch.nn as nn

class TextClassification(nn.Module):
    def __init__(self, args):
        super(TextClassification, self).__init__()

        self.args = args
        self.num_layers = args.num_layers
        self.input_embedd = args.num_words
        self.embedd_dim = args.embedding_dim
        self.dropout = nn.Dropout(0.5)
        self.num_class = 18

        self.embedd = nn.Embedding(self.input_embedd, self.embedd_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size = self.embedd_dim, hidden_size = self.embedd_dim, num_layers = self.num_layers, batch_first = True)
        self.fc1 = nn.Linear(in_features=self.embedd_dim, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=self.num_layers)

    def forward(self,input):
        h = torch.zeros((self.num_layers,input.size(0),self.embedd_dim))
        c = torch.zeros((self.num_layers,input.size(0),self.embedd_dim))

        out = self.embedd(input)

        out, (hidden, cell) = self.lstm(out, (h,c))

        out = self.dropout(out)

        out = torch.relu_(self.fc1(out[:,-1,:]))

        out = self.fc2(out)

        return out
    
        