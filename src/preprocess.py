from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

class Preprocess:
    def __init__(self, args):
        self.data_path = "data/news_categories.txt"
        self.test_size = 0.2
        self.num_words = args.num_words
        self.max_len = args.max_len

    def load_data(self):
        text = []
        label = []

        for line in open(self.data_path, encoding="utf-8"):
            words = line.strip().split()
            text.append(words[0])
            label.append(" ".join(words[1:]))
        
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(text, label, test_size=self.test_size)
    
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.Y_train)

        self.Y_train = self.label_encoder.transform(self.Y_train)
        self.Y_test = self.label_encoder.transform(self.Y_test)
    
    def Tokenize(self):
        self.tokenizer = Tokenizer(self.num_words)
        self.tokenizer.fit_on_texts(self.X_train)
    
    def sequence_to_text(self, input):
        sentence = self.tokenizer.texts_to_sequences(input)
        return pad_sequences(sentence, self.max_len)




        
        