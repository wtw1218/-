import tensorflow as tf
from tensorflow import keras
import numpy as np

# 基础配置
VOCAB_PATH = '../tmp/classify_model/cnews.vocab.txt'
MODEL_PATH = '../tmp/classify_model/my_model.h5'
SEQ_LENGTH = 600

def classify_model_loading():
    # 加载模型
    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded.")

    # 加载词表和类别
    words, word_to_id = read_vocab(VOCAB_PATH)
    categories, cat_to_id, id_to_cat = read_category()

    return model, word_to_id, id_to_cat

# 读取词表
def open_file(filename, mode='r'):
    return open(filename, mode, encoding='utf-8', errors='ignore')

def read_vocab(vocab_path):
    with open_file(vocab_path) as f:
        words = [line.strip() for line in f]
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id


# 分类标签
def read_category():
    categories = ['体育', '财经', '房产', '家居', '教育',
                  '科技', '时尚', '时政', '游戏', '娱乐']
    cat_to_id = dict(zip(categories, range(len(categories))))
    id_to_cat = dict(zip(range(len(categories)), categories))
    return categories, cat_to_id, id_to_cat

# 文本预处理（和训练时保持一致）
def text_to_id(text, word_to_id):
    return [word_to_id[x] for x in text if x in word_to_id]

def preprocess(text, word_to_id):
    ids = text_to_id(text, word_to_id)
    x_pad = keras.preprocessing.sequence.pad_sequences(
        [ids],
        maxlen=SEQ_LENGTH
    )
    return x_pad

# 推理函数
def predict_text(text, model, word_to_id, id_to_cat):
    x = preprocess(text, word_to_id)
    probs = model.predict(x)[0]

    pred_id = np.argmax(probs)
    pred_label = id_to_cat[pred_id]
    confidence = probs[pred_id]

    return {
        "text": text,
        "label": pred_label,
        "confidence": float(confidence)
    }

if __name__ == "__main__":

    text = "篮球"

    loaded_model, word_to_id, id_to_cat = classify_model_loading()

    result = predict_text(text, loaded_model, word_to_id, id_to_cat)
    print(f"预测类别：{result['label']} | 置信度：{result['confidence']:.4f}")
