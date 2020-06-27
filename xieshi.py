import math
import re
import numpy as np
import tensorflow as tf
from collections import Counter
DATA_PATH = './kecheng/poetry.txt'
MAX_LEN = 64
DISALLOWED_WORDS = ['（', '）', '(', ')', '__', '《', '》', '【', '】', '[', ']']	
poetry = []
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    lines = f.readlines()
for line in lines:
    fields = re.split(r"[:：]", line)
    if len(fields) != 2:
        continue
    content = fields[1]
    if len(content) > MAX_LEN - 2:
        continue
    if any(word in content for word in DISALLOWED_WORDS):
        continue
        
    poetry.append(content.replace('\n', '')) 

for i in range(0, 5):
    print(poetry[i])

MIN_WORD_FREQUENCY = 8
counter = Counter()
for line in poetry:
    counter.update(line)
tokens = [token for token, count in counter.items() if count >= MIN_WORD_FREQUENCY]

i = 0
for token, count in counter.items():
    if i >= 5:
        break
    print(token, "->",count)
    i += 1


tokens = ["[PAD]", "[NONE]", "[START]", "[END]"] + tokens
# 映射: 词 -> 编号
word_idx = {}
# 映射: 编号 -> 词
idx_word = {}
for idx, word in enumerate(tokens):
    word_idx[word] = idx
    idx_word[idx] = word


class Tokenizer:
    """
    分词器
    """
    def __init__(self, tokens):
        self.dict_size = len(tokens)
        # 生成映射关系
        self.token_id = {} 
        self.id_token = {} 
        for idx, word in enumerate(tokens):
            self.token_id[word] = idx
            self.id_token[idx] = word
        
        self.start_id = self.token_id["[START]"]
        self.end_id = self.token_id["[END]"]
        self.none_id = self.token_id["[NONE]"]
        self.pad_id = self.token_id["[PAD]"]

    def id_to_token(self, token_id):
        return self.id_token.get(token_id)

    def token_to_id(self, token):
        return self.token_id.get(token, self.none_id)

    def encode(self, tokens):
        token_ids = [self.start_id, ] # 起始标记
        for token in tokens:
            token_ids.append(self.token_to_id(token))
        token_ids.append(self.end_id) # 结束标记
        return token_ids

    def decode(self, token_ids):
        # 起始、结束标记
        flag_tokens = {"[START]", "[END]"}
        
        tokens = []
        for idx in token_ids:
            token = self.id_to_token(idx)
            # 跳过起始、结束标记
            if token not in flag_tokens:
                tokens.append(token)
        return tokens

tokenizer = Tokenizer(tokens)

class PoetryDataSet:
    def __init__(self, data, tokenizer, batch_size):
        # 数据集
        self.data = data
        self.total_size = len(self.data)
        self.tokenizer = tokenizer
        self.batch_size = BATCH_SIZE
        self.steps = int(math.floor(len(self.data) / self.batch_size))
    
    def pad_line(self, line, length, padding=None):
        if padding is None:
            padding = self.tokenizer.pad_id
            
        padding_length = length - len(line)
        if padding_length > 0:
            return line + [padding] * padding_length
        else:
            return line[:length]
        
    def __len__(self):
        return self.steps

    def __iter__(self):
        # 打乱数据
        np.random.shuffle(self.data)
        for start in range(0, self.total_size, self.batch_size):
            end = min(start + self.batch_size, self.total_size)
            data = self.data[start:end]
            
            max_length = max(map(len, data)) 
            
            batch_data = []
            for str_line in data:
                # 对每一行诗词进行编码、并补齐padding
                encode_line = self.tokenizer.encode(str_line)
                pad_encode_line = self.pad_line(encode_line, max_length + 2) 
                batch_data.append(pad_encode_line)

            batch_data = np.array(batch_data)
            yield batch_data[:, :-1], batch_data[:, 1:]

    def generator(self):
        while True:
            yield from self.__iter__()

BATCH_SIZE = 32
dataset = PoetryDataSet(poetry, tokenizer, BATCH_SIZE)


model = tf.keras.Sequential([
    # 词嵌入层
    tf.keras.layers.Embedding(input_dim=tokenizer.dict_size, output_dim=150),
    # 第一个LSTM层
    tf.keras.layers.LSTM(150, dropout=0.5, return_sequences=True),
    # 第二个LSTM层
    tf.keras.layers.LSTM(150, dropout=0.5, return_sequences=True),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(tokenizer.dict_size, activation='softmax')),
])

model.summary()
model.compile(
    optimizer=tf.keras.optimizers.Adam(), 
    loss=tf.keras.losses.sparse_categorical_crossentropy
)

#训练模型

model.fit(
    dataset.generator(), 
    steps_per_epoch=dataset.steps, 
    epochs=10
)

#预测单个词
token_ids = [tokenizer.token_to_id(word) for word in ["月", "亮", "弯", "弯"]]
# 进行预测
result = model.predict([token_ids ,])
print(result)


def predict(model, token_ids):
    _probas = model.predict([token_ids, ])[0, -1, 3:]
    # 按概率降序，取前100
    p_args = _probas.argsort()[-100:][::-1] # 索引
    p = _probas[p_args] # 根据索引找到具体的概率值
    p = p / sum(p) # 归一
    target_index = np.random.choice(len(p), p=p)
    return p_args[target_index] + 3



def generate_random_poem(tokenizer, model, text=""):
    token_ids = tokenizer.encode(text)[:-1]
    while len(token_ids) < MAX_LEN:
        # 预测词的编号
        target = predict(model, token_ids)
        # 保存结果
        token_ids.append(target)
        # 到达END
        if target == tokenizer.end_id: 
            break
        
    return "".join(tokenizer.decode(token_ids))
for i in range(3):
    print(generate_random_poem(tokenizer, model))

print(generate_random_poem(tokenizer, model, "春眠不觉晓，"))


#藏头诗
def generate_acrostic_poem(tokenizer, model, heads):
    token_ids = [tokenizer.start_id, ]
    punctuation_ids = {tokenizer.token_to_id("，"), tokenizer.token_to_id("。")}
    content = []
    for head in heads:
        content.append(head)
        token_ids.append(tokenizer.token_to_id(head))
        target = -1
        while target not in punctuation_ids: 
            # 预测词的编号
            target = predict(model, token_ids)
            # 因为可能预测到END，所以加个判断
            if target > 3:
                token_ids.append(target)
                content.append(tokenizer.id_to_token(target))

    return "".join(content)
print(generate_acrostic_poem(tokenizer, model, heads="写个代码"))
