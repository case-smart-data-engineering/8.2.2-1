# BERT-BiLSTM-Softmax模型进行REL实体关系抽取

### 文件清单
- mains：NER和REL的训练启动文件，二者分开执行，先进行trainer_ner后进行trainer_rel。
- data_loader：存放的是数据处理器，完成对数据的预处理-->BIO格式。
- data：存放的是train、test和dev的数据集。
- config_utils：存放的是NER和REL的网络模型参数。
- models：保存的是NER和REL训练的checkpoint。
- modules：保存的是NER和REL的网络结构。
- solution：运行入口，test函数先运行实体识别，然后运行关系抽取。


### 运行步骤
1. 下载bert模型，采用的是bert-base-chinese(pytorch版本)，下载连接：https://huggingface.co/bert-base-chinese/tree/main 。
2. 把下载的bert预训练模型(pytorch_model.bin文件)上传到"1_算法示例/bert-base-chinese"目录下即可运行data_loader/process_rel.py文件。
3. `下载`已训练好的模型文件并`上传`到指定路径下即可，用于直接运行solution.py文件进行测试。
   1. xx_ner.pth文件，放置于路径models/sequence_labeling/ 。
   2. xx_rel.pth文件，放置于路径models/rel_cls/ 。
   3. 下载路径：https://github.com/KaiserLord/bigFiles/tree/master/models 。

> 对于数据部分，由于github的限制，以及对想要测试模型的读者，github的项目中存放的是原始数据集的切片，这样便于在线化运行。如果有想要全部数据集的读者，可以从[https://pan.baidu.com/s/1XK3v6BQlnsvhGxgg-71IpA]下载json文件，密码'nlp0'，放置于`1_算法示例/data/` 。

### 模型训练
在mains目录下，分别运行trainer_ner.py和trainer_rel.py进行训练。

## 项目说明
项目分两部分：
- 命名实体识别部分使用的是BiLSTM+CRF。
- 实体关系抽取使用的是Bert+Softmax进行关系分类。

### 数据集说明
百度的DUIE数据集是业界规模最大的中文信息抽取数据集。它包含了43万三元组数据、21万中文句子。
句子的平均长度为54，每句话中的三元组数量的平均值为2.1。
下面是一个样本：
{"text": "据了解，《小姨多鹤》主要在大连和丹东大梨树影视城取景，是导演安建继《北风那个吹》之后拍摄的又一部极具东北文化气息的作品", 
  "spo_list": [{
  "predicate": "导演",
  "object_type": "人物",
  "subject_type": "影视作品",
  "object": "安建",
  "subject": "小姨多鹤"
  }, {
  "predicate": "导演",
  "object_type": "人物",
  "subject_type": "影视作品",
  "object": "安建",
  "subject": "北风那个吹"
  }]
}

## 运行结果
在训练完后会在`1_算法示例/models/`路径下保存训练结果，然后运行demo.py文件，会得到关系类别的预测结果

## 备注
如果是在命令行上运行代码，需要按照使用指南的顺序进入`正确的目录`才可运行成功。