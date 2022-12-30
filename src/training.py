import torch
from preprocess import Preprocess
from model import TextClassification
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

SEED = 2019
torch.manual_seed(SEED)

class DataMapper(Dataset):
    def __init__(self,x ,y):
        self.x = x
        self.y = y
    
    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]    

class execute:
    def __init__(self,args):
        self.__init_data__(args)

        self.batch_size = args.batch_size
    
    def __init_data__(self):
        self.preprocess = Preprocess()
        self.preprocess.load_data()
        self.preprocess.Tokenize()

        raw_train_data = self.preprocess.X_train
        raw_test_data = self.preprocess.X_test
        
        self.x_train = self.preprocess.sequence_to_text(raw_train_data)
        self.x_test = self.preprocess.sequence_to_text(raw_test_data)

        self.y_train = self.preprocess.Y_train
        self.y_test = self.preprocess.Y_test

    def train(self):
        train = DataMapper(self.x_train, self.y_train)
        test = DataMapper(self.x_test, self.y_test)

        load_train = DataLoader(train, batch_size = self.batch_size)
        load_test = DataLoader(test, batch_size=self.batch_size)
        
        




            