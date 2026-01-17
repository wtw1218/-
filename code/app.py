import json
from flask import Flask, render_template, request, jsonify, session
from doubao.llm_call import doubao_call

from classify_feature import classify_model_loading, predict_text
from sentiment_feature import sentiment_model_loading, predict_sentiment
from translate_feature import translate_model_loading, translate

# Flask 配置
app = Flask(__name__, static_url_path='/static')
app.secret_key = "chat-secret-key"

# 模型加载
classify_model, classify_word_to_id, classify_id_to_cat = classify_model_loading()
sentiment_model, sentiment_word_to_id = sentiment_model_loading()
translate_inp_lang, translate_targ_lang, translate_encoder, translate_decoder, translate_max_length_inp, translate_max_length_targ, translate_units = translate_model_loading()

# Router Prompt
ROUTER_PROMPT = """
你是一个任务路由器，需要从用户输入中判断任务类型，
并抽取真正需要被处理的文本内容。

请严格按照 JSON 格式输出，不要包含任何多余文字。

任务类型只能是以下 4 种之一：
- chat
- sentiment
- translate
- classify

如果是 chat：
- text 为空字符串

如果是 sentiment / translate / classify：
- text 只包含需要处理的文本

用户输入：
\"\"\"{user_input}\"\"\"

输出 JSON：
{{
  "task": "...",
  "text": "..."
}}
"""

# 对话路由选择
def chat_router(user_input: str):
    prompt = ROUTER_PROMPT.format(user_input=user_input)
    resp = doubao_call(prompt)
    return json.loads(resp.strip())


# 多轮对话历史维护
def chat_with_history(user_input, history):

    messages = []
    for h in history:
        messages.append(f"{h['role']}: {h['content']}")

    messages.append(f"user: {user_input}")

    prompt = (
        "你是一个中文对话助手，请基于上下文自然回复。\n\n"
        + "\n".join(messages)
        + "\nassistant:"
    )

    reply = doubao_call(prompt)

    return reply

# 核心聊天逻辑
def chat(sentence: str):

    if "history" not in session:
        session["history"] = []

    history = session["history"]

    route = chat_router(sentence)
    task = route["task"]
    process_text = route["text"]

    if task == "sentiment":
        result = "文本情感分析结果：" + predict_sentiment(
            process_text, sentiment_model, sentiment_word_to_id
        )["sentiment"]

    elif task == "translate":
        result = "文本翻译结果：" + translate(
            process_text,
            translate_inp_lang,
            translate_targ_lang,
            translate_encoder,
            translate_decoder,
            translate_max_length_inp,
            translate_max_length_targ,
            translate_units
        )

    elif task == "classify":
        result = "文本分类结果：" + predict_text(
            process_text, classify_model, classify_word_to_id, classify_id_to_cat
        )["label"]

    else:
        # 只有 chat 才吃历史
        result = chat_with_history(sentence, history)

    # 更新历史
    history.append({"role": "user", "content": sentence})
    history.append({"role": "assistant", "content": result})

    # 控制历史长度
    session["history"] = history[-10:]

    return result

# 接口
@app.route('/message', methods=['POST'])
def reply():
    req_msg = request.form['msg']
    res_msg = chat(req_msg)
    res_msg = res_msg.replace('_UNK', '^_^').strip()
    if not res_msg:
        res_msg = '我们来聊聊天吧'
    return jsonify({'text': res_msg})

@app.route("/")
def index():
    session.clear()
    return render_template('index.html')

# 启动
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8808)
