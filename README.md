# Phân loại văn bản tiếng Việt

Đây là repo phân loại văn bản tiếng Việt

## Tính năng

- Dữ liệu tiếng Việt
- Sử mạng kiến trúc mạng LSTM

## Cài đặt

Yêu cầu [python](https://www.python.org/) >= 3.6.

Cài đặt các thư viện cần thiết để khởi chạy server.

```sh
git clone https://github.com/hieunguyenquoc/vietnam-text-classification.git
cd text-classification-pytorch
pip install -r requirements.txt
python ./src/main.py
```

Huấn luyện

```sh
python ./src/training.py
```