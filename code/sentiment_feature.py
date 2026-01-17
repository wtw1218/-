import tensorflow as tf
import pandas as pd
import jieba
from tensorflow.keras.preprocessing import sequence

# 基础配置
MODEL_PATH = "../tmp/sentiment_model/lstm_sentiment.h5"
VOCAB_PATH = "../tmp/sentiment_model/sentiment_vocab.csv"
MAXLEN = 50

# 加载模型，词典
def sentiment_model_loading():

    print("Loading sentiment model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded.")
    dicts = pd.read_csv(VOCAB_PATH, index_col=0)
    word_to_id = dict(zip(dicts.index, dicts["id"]))
    return model, word_to_id

# 文本预处理
def preprocess(text, word_to_id):
    words = list(jieba.cut(str(text)))
    sent = [word_to_id[w] for w in words if w in word_to_id]
    x_pad = sequence.pad_sequences([sent], maxlen=MAXLEN)
    return x_pad

# 推理函数
def predict_sentiment(text, model, word_to_id):
    x = preprocess(text, word_to_id)
    prob = model.predict(x)[0][0]

    label = "正向" if prob >= 0.5 else "负向"

    return {
        "text": text,
        "sentiment": label,
        "confidence": float(prob)
    }

# 测试
if __name__ == "__main__":
    text = "你是个好人"

    model, word_to_id = sentiment_model_loading()

    result = predict_sentiment(text, model, word_to_id)
    print(
        f"情感：{result['sentiment']} | 置信度：{result['confidence']:.4f}"
    )
