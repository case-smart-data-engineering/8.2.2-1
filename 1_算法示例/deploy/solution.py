import torch
import sys

sys.path.append('/workspace/8.2.2-1/1_算法示例/')
from modules.model_ner import SeqLabel
from modules.model_rel import AttBiLSTM
from config_utils.config_rel import ConfigRel, USE_CUDA
from config_utils.config_ner import ConfigNer, USE_CUDA

from data_loader.process_ner import ModelDataPreparation
from data_loader.process_rel import DataPreparationRel

from mains import trainer_ner, trainer_rel
import json
from transformers import BertForSequenceClassification
from transformers import logging
logging.set_verbosity_warning()
logging.set_verbosity_error()

def get_entities(pred_ner, text):
    token_types = [[] for _ in range(len(pred_ner))]
    entities = [[] for _ in range(len(pred_ner))]
    for i in range(len(pred_ner)):
        token_type = []
        entity = []
        j = 0
        word_begin = False
        while j < len(pred_ner[i]):
            if pred_ner[i][j][0] == 'B':
                if word_begin:
                    token_type = []  # 防止多个B出现在一起
                    entity = []
                token_type.append(pred_ner[i][j])
                entity.append(text[i][j])
                word_begin = True
            elif pred_ner[i][j][0] == 'I':
                if word_begin:
                    token_type.append(pred_ner[i][j])
                    entity.append(text[i][j])
            else:
                if word_begin:
                    token_types[i].append(''.join(token_type))
                    token_type = []
                    entities[i].append(''.join(entity))
                    entity = []
                word_begin = False
            j += 1
    return token_types, entities

def test():
    print("命名实体识别：")
    test_path = '/workspace/8.2.2-1/1_算法示例/data/test_data.json'
    PATH_NER =  '/workspace/8.2.2-1/1_算法示例/models/sequence_labeling/2m-f0.00n26393.14ccks2019_ner.pth'
    
    config_ner = ConfigNer()
    ner_model = SeqLabel(config_ner)
    ner_model_dict = torch.load(PATH_NER, map_location ='cpu')
    ner_model.load_state_dict(ner_model_dict['state_dict'])
    
    ner_data_process = ModelDataPreparation(config_ner)
    _, _, test_loader = ner_data_process.get_train_dev_data(path_test=test_path)
    

    trainerNer = trainer_ner.Trainer(ner_model, config_ner, test_dataset=test_loader)
    pred_ner = trainerNer.predict()
    text = None
    for data_item in test_loader:
        text = data_item['text']
    token_types, entities = get_entities(pred_ner, text)
    print(text)
    print('识别出来的实体如下:')
    print(entities)
    
    rel_list = []
    # 把从test json文件中选取的三含数据中抽取出来的实体，按格式写入rel_predict.json文件中
    with open('/workspace/8.2.2-1/1_算法示例/deploy/rel_predict.json', 'w', encoding='utf-8') as f:
        for i in range(len(pred_ner)):
            texti = text[i]
            for j in range(len(entities[i])): # entities是二维数组
                for k in range(len(entities[i])):
                    if j == k:
                        continue
                    # rel_list.append({"text":texti, "spo_list":{"subject": entities[i][j], "object": entities[i][k]}})
                    subject = entities[i][j]
                    object = entities[i][k]
                    relation = ''
                    sentence_cls = ''.join([subject, object, texti])
                    rel_list.append({"sentence_cls":sentence_cls, 'relation': relation, 'text': texti, "subject": subject, "object": object})
        json.dump(rel_list, f, ensure_ascii=False)

    print("实体关系抽取：")
    # # 加载模型参数
    PATH_REL = '/workspace/8.2.2-1/1_算法示例/models/rel_cls/1m-acc0.75ccks2019_rel.pth'
    
    config_rel = ConfigRel()
    # config_rel.batch_size = 8
    rel_model = BertForSequenceClassification.from_pretrained('/workspace/8.2.2-1/1_算法示例/bert-base-chinese', num_labels=config_rel.num_relations)
    rel_model_dict = torch.load(PATH_REL, map_location ='cpu')
    # print(rel_model_dict.keys())
    rel_model.load_state_dict(rel_model_dict['state_dict'], False)
    rel_test_path = '/workspace/8.2.2-1/1_算法示例/data/test_data.json' 

    rel_data_process = DataPreparationRel(config_rel)
    _, _, test_loader = rel_data_process.get_train_dev_data(path_test=rel_test_path) # 测试数据

    trainREL = trainer_rel.Trainer(rel_model, config_rel, test_dataset=test_loader)
    rel_pred = trainREL.bert_predict()
    return rel_pred

if __name__ == '__main__':
    res = test()
    print(res)
