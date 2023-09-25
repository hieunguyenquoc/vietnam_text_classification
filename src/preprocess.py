from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import re
from underthesea import word_tokenize

class Preprocess:
    def __init__(self, args):
        self.data_path = "data/news_categories.txt"
        self.test_size = 0.2
        self.num_words = args.num_words
        self.max_len = args.max_len
        self.stopwords_path = "stopwords.txt"

    def load_data(self):
        text = []
        label = []

        for line in open(self.data_path, encoding="utf-8"):
            words = line.strip().split()
            label.append(words[0])
            text.append(" ".join(words[1:]))
        
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(text, label, test_size=self.test_size)
    
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.Y_train)

        self.Y_train = self.label_encoder.transform(self.Y_train)
        self.Y_test = self.label_encoder.transform(self.Y_test)

    def preprocess_data(self,strings):
        strings = strings.lower()
        strings = strings.split()
        f = open(self.stopwords_path, 'r', encoding="utf-8")  
        stopwords = f.readlines()
        stop_words = [s.replace("\n", '') for s in stopwords]
        #print("mid: ", stop_words)
        doc_words = []
        #### YOUR CODE HERE ####
        
        for word in strings:
            if word not in stop_words:
                doc_words.append(word)

        #### END YOUR CODE #####
        doc_str = ' '.join(doc_words).strip()
        
        text = re.sub(r'[^\s\wáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ_]',' ',doc_str)
        
        text = re.sub(r'\s+', ' ', text).strip()

        return text
        
    def Tokenize(self):
        self.tokenizer = Tokenizer(self.num_words)
        self.tokenizer.fit_on_texts(self.X_train)
    
    def sequence_to_text(self, input):
        sentence = [word_tokenize(text, format="text") for text in input]
        sentence = self.tokenizer.texts_to_sequences(sentence)
        return pad_sequences(sentence, self.max_len)




        
        