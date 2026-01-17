import tensorflow as tf
import pickle
import re

# 配置路径
CHECKPOINT_DIR = "../tmp/translate_model/training_checkpoints"
INP_LANG_PATH = "../tmp/translate_model/inp_lang.pkl"
TARG_LANG_PATH = "../tmp/translate_model/targ_lang.pkl"
CONFIG_PATH = "../tmp/translate_model/translate_config.pkl"

# 模型定义
class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units):
        super().__init__()
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(
            self.enc_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        hidden_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(
            self.W1(values) + self.W2(hidden_with_time_axis)
        ))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units):
        super().__init__()
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(
            self.dec_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )
        self.fc = tf.keras.layers.Dense(vocab_size)
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        context_vector, attention_weights = self.attention(hidden, enc_output)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)
        return x, state, attention_weights

# 文本预处理
def preprocess_sentence(w):
    w = re.sub(r'([?.!,])', r' \1 ', w)
    w = re.sub(r'\s+', ' ', w)
    w = '<start> ' + w + ' <end>'
    return w

def translate_model_loading():
    # 加载 tokenizer & config
    with open(INP_LANG_PATH, "rb") as f:
        inp_lang = pickle.load(f)

    with open(TARG_LANG_PATH, "rb") as f:
        targ_lang = pickle.load(f)

    with open(CONFIG_PATH, "rb") as f:
        config = pickle.load(f)

    max_length_inp = config["max_length_inp"]
    max_length_targ = config["max_length_targ"]
    units = config["units"]

    vocab_inp_size = len(inp_lang.word_index) + 1
    vocab_tar_size = len(targ_lang.word_index) + 1
    embedding_dim = 256

    # 加载模型 + 权重
    encoder = Encoder(vocab_inp_size, embedding_dim, units)
    decoder = Decoder(vocab_tar_size, embedding_dim, units)

    checkpoint = tf.train.Checkpoint(encoder=encoder, decoder=decoder)
    checkpoint.restore(tf.train.latest_checkpoint(CHECKPOINT_DIR)).expect_partial()

    print("Translation model loaded.")

    return inp_lang, targ_lang, encoder, decoder, max_length_inp, max_length_targ, units

# 推理函数
def translate(sentence, inp_lang, targ_lang, encoder, decoder, max_length_inp, max_length_targ, units):
    sentence = preprocess_sentence(sentence)
    inputs = [inp_lang.word_index.get(i, 0) for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences(
        [inputs], maxlen=max_length_inp, padding='post'
    )
    inputs = tf.convert_to_tensor(inputs)

    result_words = []
    hidden = tf.zeros((1, units))
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)

    for _ in range(max_length_targ):
        predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_out)
        predicted_id = tf.argmax(predictions[0]).numpy()
        word = targ_lang.index_word.get(predicted_id, '')

        if word == '<end>':
            break

        result_words.append(word)
        dec_input = tf.expand_dims([predicted_id], 0)

    return ' '.join(result_words)


# 测试
if __name__ == "__main__":

    inp_lang, targ_lang, encoder, decoder, max_length_inp, max_length_targ, units = translate_model_loading()

    print(translate("我生病了。", inp_lang, targ_lang, encoder, decoder, max_length_inp, max_length_targ, units))
    print(translate("为什么不？", inp_lang, targ_lang, encoder, decoder, max_length_inp, max_length_targ, units))
    print(translate("让我一个人呆会儿。", inp_lang, targ_lang, encoder, decoder, max_length_inp, max_length_targ, units))
    print(translate("打电话回家！", inp_lang, targ_lang, encoder, decoder, max_length_inp, max_length_targ, units))
    print(translate("我了解你。", inp_lang, targ_lang, encoder, decoder, max_length_inp, max_length_targ, units))