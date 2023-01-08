from model import TextClassification
import torch
from preprocess import Preprocess
from parser_param import parameter_parser
import torch.nn.functional as F
from fastapi import FastAPI
import uvicorn

class Inference:
    def __init__(self):
        args = parameter_parser()
        self.model = TextClassification(args)
        self.model = torch.load("model/viet_nam_classification.ckpt")
        self.model.eval()
        self.preprocess = Preprocess(args)
        self.preprocess.load_data()
        self.preprocess.Tokenize()

    def predict(self, sentence):
        sentence_preprosess = self.preprocess.preprocess_data(sentence)
        sentence_predict = self.preprocess.sequence_to_text(sentence_preprosess)
        sentence_tensor = torch.from_numpy(sentence_predict)
        sentence_final = sentence_tensor.type(torch.LongTensor)

        with torch.no_grad():
            y_pred = self.model(sentence_final)
            y_pred = F.softmax(y_pred).numpy()

            y_pred = y_pred.argmax(axis=1)
            y_pred = self.preprocess.label_encoder.classes_[y_pred]
        
        return y_pred[0]

app = FastAPI()

@app.post("/text_classification")
def text_classification(sentence : str):
    result = Inference()
    result_final = result.predict(sentence)
    return result_final

if __name__ == "__main__":
    uvicorn.run(app, host = "0.0.0.0", port = 8000)