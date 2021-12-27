# AI-Poet
一款基于深度学习的 AI 作诗系统

**id: 519021910861**
**name: 徐惠东**

（个人单独组队）

## 简介

近年来，人工智能渗透了我们生活的方方面面，给我们的衣食住行都带来了不少便利。纵观当今的人工智能在交通管理、环境治理、数字能源和社会治理的应用，无论是推荐系统还是异常检测，他们都在为人们的生存和生活添砖加瓦。

然而，“生活不止眼前的苟且，还有**诗**和远方”。我们有理由相信，当人们的基本生活需求已经满足后，**作诗读诗**将会是人工智能的高层次应用。正如中国在盛唐时期古诗遍地开花，在不久的将来，人工智能写诗也必将掀起一波浪潮。

因此，AI-Poet 参考《Pytorch入门与实践》教程和 [GitLab 上开源代码框架](https://gitee.com/wannabe-9/LSTM_poem1)，采用[**循环神经网络 (RNN) **](https://zh.wikipedia.org/wiki/%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C)中的[**长短期记忆 (LSTM) **](https://zh.wikipedia.org/wiki/%E9%95%B7%E7%9F%AD%E6%9C%9F%E8%A8%98%E6%86%B6)模型进行训练，最终效果可以支持**首句续写**和**藏头诗**两种功能，并且实现了基本的音律和意境。

**关键词**
深度学习  循环神经网络  长短期记忆  预训练模型

## 环境配置

最初试图使用 _MacBook Pro 2017_ 进行训练，然而因为是因特尔显卡没办法直接使用 _cuda_，所以只能使用 CPU 导致训练速度极慢。经过粗略计算，跑完一遍模型需要约 15 天时间，因此不在本机跑模型。

之后尝试过阿里云租借服务器、百度飞桨平台等，但都觉得过于复杂。最终选用谷歌的 Colab 平台进行训练，所分配到的 GPU 为 _Tesla P100_，训练一次模型时间约为 3 小时。

![GPU](./img/GPU.png)

## 代码实现

### 模型接口
主要关注 LSTM 层数，epoch 和 batch_size 参数。

**LSTM 层数**
共三层，分别为 embedding 层、lstm 层和 linear 层。

**epoch**
向前和向后传播中所有批次的单次训练迭代，即训练过程中全部样本数据将被“轮”多少次。

**batch_size**
基本上现在的梯度下降都是基于 mini-batch 的，每次训练使用 batch_size 个数据进行参数寻优，一批中的数据共同决定了本次梯度的方向。


```python
class Config(object):
    num_layers = 3  # LSTM层数
		...
    lr = 1e-3
    weight_decay = 1e-4
    use_gpu = True
    epoch = 30
    batch_size = 25
    maxlen = 125  # 超过这个长度的之后字被丢弃，小于这个长度的在前面补空格
    plot_every = 200  # 每20个batch 可视化一次
    max_gen_len = 200  # 生成诗歌最长长度
    ...
    embedding_dim = 256
    hidden_dim = 512
    ...
```

### 数据集

使用整理好的 **numpy** 格式的[开源数据集](http://pytorch-1252820389.cosbj.myqcloud.com/tang_199.pth)，其中包含唐诗共 57580 首 * 125 字，不足和超出 125 字的都已经被补全或者截断。

```Python
# 处理数据
datas = np.load("/content/drive/MyDrive/AI-Poet/tang.npz", allow_pickle=True)
data = datas['data']
ix2word = datas['ix2word'].item()
word2ix = datas['word2ix'].item()
data = t.from_numpy(data)
dataloader = DataLoader(data,
                        batch_size=Config.batch_size,
                        shuffle=True,
                        num_workers=2)
```

**细节解释**
1. data 是 numpy 数组，共 57580 首 * 125 字。
2. word2ix 和 ix2word 都是字典类型，用于字符和序号的映射。

### LSTM 循环神经网络

LSTM 是一种特殊的 RNN，能够解决长序列训练过程中的梯度消失和梯度爆炸问题，相比于 RNN 只有一个传递状态 $h^t$，LSTM 有两个传输状态分别是 $c^t$ (cell state) 和 $h^t$ (hidden state)。

其中对于传递下去的 $c^t$ 改变得较慢，通常输出的 $c^t$ 是上一个状态传过来的 $c^{t - 1}$ 加上一些数值，而 $h^t$ 则在不同节点下有较大区别。本模型便采用了 LSTM 模型进行训练。

```Python
class PoetryModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(PoetryModel, self).__init__()
        self.hidden_dim = hidden_dim
        # 词向量层，词表大小 * 向量维度
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # 网络主要结构
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=Config.num_layers)
        # 进行分类
        self.linear = nn.Linear(self.hidden_dim, vocab_size)

    def forward(self, input, hidden=None):
        seq_len, batch_size = input.size()
        # print(input.shape)
        if hidden is None:
            h_0 = input.data.new(Config.num_layers, batch_size, self.hidden_dim).fill_(0).float()
            c_0 = input.data.new(Config.num_layers, batch_size, self.hidden_dim).fill_(0).float()
        else:
            h_0, c_0 = hidden
        # 输入 序列长度 * batch(每个汉字是一个数字下标)，
        # 输出 序列长度 * batch * 向量维度
        embeds = self.embeddings(input)
        # 输出hidden的大小： 序列长度 * batch * hidden_dim
        output, hidden = self.lstm(embeds, (h_0, c_0))
        output = self.linear(output.view(seq_len * batch_size, -1))
        return output, hidden
```

**细节解释**

![LSTM](./img/LSTM.png)

将数据集作为喂给模型的作为 _input_，先经过 _embedding_ 预处理得到 _embeds_ 层，然后经过 _LSTM_ 进行训练得到 _hidden_ 层和 _output_ 层，最后经过 _Linear_ 层判别，然后反向传播并循环训练即可。

### 训练方法
使用 GPU 进行训练，每次将数据输入进 LSTM 网络进行前向传播训练，然后使用误差反向传播进行修正，每隔一定数据量进行一次可视化，不断迭代更新。
```python
loss_data = []
# 进行训练并画图
def train():
  f = open(Config.result_path, 'w')
  for epoch in range(basic_start, Config.epoch):
      loss_meter.reset()
      for li, data_ in tqdm.tqdm(enumerate(dataloader)):
          # 将数据转置并复制一份
          data_ = data_.long().transpose(1, 0).contiguous()
          # 注意这里，也转移到了计算设备上
          data_ = data_.to(device)
          Configimizer.zero_grad()
          # n个句子，前n-1句作为inout，后n-1句作为label，二者一一对应
          # 经过 LSTM 网络进行前向传播
          input_, target = data_[:-1, :], data_[1:, :]
          output, _ = model(input_)
          # 误差反向传播
          loss = criterion(output, target.view(-1))
          loss.backward()
          Configimizer.step()
          loss_meter.add(loss.item())
          # 存储 loss 数据，方便之后画图
          loss_data.append(loss)
          # 进行可视化
          if (1 + li) % Config.plot_every == 0:
              print("训练损失为%s\n" % (str(loss_meter.mean)))
              ...
train()
```

### 诗句生成

**首句生成模式**

优先使用风格前缀生成隐藏层，并结合用户输入的首句喂给预训练模型生成下一句，再使用生成的下一句作为下一次迭代的输入，不断迭代直至达到最大生成字数或遇到终止符 \<EOP\> 为止。

```python
# 给定首句生成诗歌
def generate(model, start_words, ix2word, word2ix, prefix_words=None):
		...   
    # 若有风格前缀，则先用风格前缀生成hidden
    if prefix_words:
        for word in prefix_words:
            output, hidden = model(input, hidden)
            input = input.data.new([word2ix[word]]).view(1, 1)

    # 开始真正生成诗句，如果没有使用风格前缀，则hidden = None，input = <START>
    # 否则，input就是风格前缀的最后一个词语，hidden也是生成出来的
    for i in range(Config.max_gen_len):
        output, hidden = model(input, hidden)
        if i < start_words_len:
            ...
            input = input.data.new([word2ix[w]]).view(1, 1)
        else:
            ...
            input = input.data.new([top_index]).view(1, 1)
        if w == '<EOP>':
            del results[-1]
            break
    return results
```

**藏头诗模式**

优先使用风格前缀生成隐藏层，并结合用户输入每次喂给模型一个字作为开头并续写，迭代更新至用户输入用完为止。

```python
# 生成藏头诗
def gen_acrostic(model, start_words, ix2word, word2ix, prefix_words=None):
    input = (t.Tensor([word2ix['<START>']]).view(1, 1).long())
		...
    # 存在风格前缀，则生成hidden
    if prefix_words:
        for word in prefix_words:
            output, hidden = model(input, hidden)
            input = (input.data.new([word2ix[word]])).view(1, 1)

    # 开始生成诗句
    for i in range(Config.max_gen_len):
        output, hidden = model(input, hidden)
        top_index = output.data[0].topk(1)[1][0].item()
        w = ix2word[top_index]
        # 说明上个字是句末
        if pre_word in {'。', '，', '?', '！', '<START>'}:
            ...
        else:
            input = (input.data.new([top_index])).view(1, 1)
        result.append(w)
        pre_word = w
    return result
print("Define usage successfully\n")
```
## 运行展示

```python
...
def userTest():
    print("正在初始化......")
    datas = np.load('/'.join([Config.data_path, Config.pickle_path]), allow_pickle=True)
    data = datas['data']
    ix2word = datas['ix2word'].item()
    word2ix = datas['word2ix'].item()
    model = PoetryModel(len(ix2word), Config.embedding_dim, Config.hidden_dim)
    model.load_state_dict(t.load(Config.model_path, 'cpu'))
    if Config.use_gpu:
        model.to(t.device('cuda'))
    print("初始化完成！\n")
    while True:
        print("欢迎使用唐诗生成器，\n"
              "输入1 进入首句生成模式\n"
              "输入2 进入藏头诗生成模式\n")
        mode = int(input())
        if mode == 1:
            print("请输入您想要的诗歌首句，可以是五言或七言")
            start_words = str(input())
            gen_poetry = ''.join(generate(model, start_words, ix2word, word2ix))
            print("生成的诗句如下：%s\n" % (gen_poetry))
            f.write("首句生成模式: %s\n %s\n" % (start_words, gen_poetry))
        elif mode == 2:
            print("请输入您想要的诗歌藏头部分，不超过16个字，最好是偶数")
            start_words = str(input())
            gen_poetry = ''.join(gen_acrostic(model, start_words, ix2word, word2ix))
            print("生成的诗句如下：%s\n" % (gen_poetry))
            f.write("藏头诗模式: %s\n %s\n" % (start_words, gen_poetry))
userTest()
```

**使用方法**
使用 Colab 打开项目，在 _AI-Poet.ipynb_ 中 _User Test_ 部分点击运行，根据提示输入 **1（首句生成）**或者 **2（藏头诗）**来选择生成诗句模式。
若是**首句生成**模式，则需再输入诗歌首句。若是**藏头诗**模式，则需输入诗歌藏头部分。

**成果展示**

## 项目总结

在一个学期的《人工智能》选修课中，我学习到了诸如机器学习、深度学习、自然语言处理等多种人工智能相关技术，也逐步了解到自动驾驶、智能推荐等人工智能的广大应用前景，感触颇深。因为我自身是软件工程专业高年级学生（大三），且之前对机器学习和深度学习等技术略有接触，因此这次课程大作业选择了单人组队，这也大大增加了我许多选题和解题的灵活性。

在我看来，AI 的各种技术目前都还在起步阶段，相对成熟的推荐系统也仅仅是通过有限的用户浏览历史来进行简单的预测，而大数据时代的到来不仅意味着个人数据的广泛搜集和使用，更为计算机的算力提升和相应模型复杂度增加提供了契机。在未来，AI 必然会更进一步，做的更好，我也希望能够参与其中。

感谢《人工智能》课程 孔令和老师、许岩岩老师的精彩授课和悉心指导，段勇帅助教也为我提供了许多帮助。





