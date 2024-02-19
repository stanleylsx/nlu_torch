# -*- coding: utf-8 -*-
# @Author : Stanley
# @Email : gzlishouxian@gmail.com
# @File : data.py
# @Software: VSCode
from transformers import BertTokenizerFast
import torch
import numpy as np


class DataManager:
    def __init__(self, configs, logger):
        self.logger = logger
        self.configs = configs
        self.train_file = self.configs['train_file']
        self.dev_file = self.configs['dev_file']
        self.batch_size = configs['batch_size']
        self.max_sequence_length = configs['max_sequence_length']

        self.entity_classes = configs['entity_classes']
        self.entity_categories = {self.entity_classes[index]: index for index in range(0, len(self.entity_classes))}
        self.entity_reverse_categories = {class_id: class_name for class_name, class_id in
                                          self.entity_categories.items()}
        self.entity_num_labels = len(self.entity_reverse_categories)

        self.intent_classes = configs['intent_classes']
        self.intent_categories = {cls: index for index, cls in enumerate(self.intent_classes)}
        self.intent_reverse_classes = {str(class_id): class_name for class_name, class_id
                                       in self.intent_categories.items()}
        self.intent_num_labels = len(self.intent_classes)
        self.tokenizer = BertTokenizerFast.from_pretrained(configs['ptm'])
        self.vocab_size = len(self.tokenizer)

    def prepare_data(self, data):
        entity_results_list = []
        entity_vectors = []
        intent_labels = []
        texts = [item.get('text') for item in data]
        token_results = self.tokenizer.batch_encode_plus(texts, padding=True, return_tensors='pt')
        token_ids_list = token_results.get('input_ids')
        token_length = token_ids_list.size(1)
        attention_list = token_results.get('attention_mask')
        for item in zip(data, texts):
            data = item[0]
            text = item[1]
            intent_label = data.get('intent')
            intent_labels.append(self.intent_categories[intent_label])
            entity_results = {}

            if self.configs['slot_model'] == 'bp':
                entity_vector = np.zeros((token_length, len(self.entity_categories), 2))
            else:
                entity_vector = np.zeros((len(self.entity_categories), token_length, token_length))

            for entity in data.get('entities'):
                start_idx = entity['start_idx']
                end_idx = entity['end_idx']
                type_class = entity['type']
                token2char_span_mapping = self.tokenizer(text,
                                                         max_length=self.max_sequence_length,
                                                         return_offsets_mapping=True,
                                                         add_special_tokens=True,
                                                         padding=True,
                                                         truncation=True)['offset_mapping']
                start_mapping = {j[0]: i for i, j in enumerate(token2char_span_mapping) if j != (0, 0)}
                end_mapping = {j[-1] - 1: i for i, j in enumerate(token2char_span_mapping) if j != (0, 0)}
                if start_idx in start_mapping and end_idx in end_mapping:
                    class_id = self.entity_categories[type_class]
                    entity_results.setdefault(class_id, set()).add(entity['entity'])
                    start_in_tokens = start_mapping[start_idx]
                    end_in_tokens = end_mapping[end_idx]
                    if self.configs['slot_model'] == 'bp':
                        entity_vector[start_in_tokens, class_id, 0] = 1
                        entity_vector[end_in_tokens, class_id, 1] = 1
                    else:
                        entity_vector[class_id, start_in_tokens, end_in_tokens] = 1
            entity_results_list.append(entity_results)
            entity_vectors.append(entity_vector)
        entity_vectors = torch.tensor(np.array(entity_vectors))
        intent_labels = torch.tensor(intent_labels)
        return texts, token_ids_list, attention_list, intent_labels, entity_results_list, entity_vectors

    def extract_entities(self, text, model_output, inference=False):
        """
        从验证集中预测到相关实体
        """
        predict_results = {}
        token2char_span_mapping = self.tokenizer(text, return_offsets_mapping=True,
                                                 max_length=self.max_sequence_length,
                                                 truncation=True)['offset_mapping']
        start_mapping = {i: j[0] for i, j in enumerate(token2char_span_mapping) if j != (0, 0)}
        end_mapping = {i: j[-1] - 1 for i, j in enumerate(token2char_span_mapping) if j != (0, 0)}
        if self.configs['slot_model'] == 'bp':
            model_output = torch.sigmoid(model_output)
            decision_threshold = float(self.configs['decision_threshold'])
            start = np.where(model_output[:, :, 0] > decision_threshold)
            end = np.where(model_output[:, :, 1] > decision_threshold)
            for _start, predicate1 in zip(*start):
                for _end, predicate2 in zip(*end):
                    if _start <= _end and predicate1 == predicate2:
                        if _start in start_mapping and _end in end_mapping:
                            start_in_text = start_mapping[_start]
                            end_in_text = end_mapping[_end]
                            entity_text = text[start_in_text: end_in_text + 1]
                            if inference:
                                predict_results.setdefault(predicate1, []).append(
                                    {'entity': entity_text, 'entity_loc': [start_in_text, end_in_text]})
                            else:
                                predict_results.setdefault(predicate1, set()).add(entity_text)
                        break
        else:
            for class_id, start, end in zip(*np.where(model_output > 0)):
                if start <= end:
                    if start in start_mapping and end in end_mapping:
                        start_in_text = start_mapping[start]
                        end_in_text = end_mapping[end]
                        entity_text = text[start_in_text: end_in_text + 1]
                        if inference:
                            predict_results.setdefault(class_id, []).append(
                                {'entity': entity_text, 'entity_loc': [start_in_text, end_in_text]})
                        else:
                            predict_results.setdefault(class_id, set()).add(entity_text)
        return predict_results

    def prepare_single_sentence(self, sentence):
        token_results = self.tokenizer(sentence)
        token_ids = token_results.get('input_ids')
        attention_mask = token_results.get('attention_mask')
        token_ids = torch.unsqueeze(torch.LongTensor(token_ids), 0)
        attention_mask = torch.unsqueeze(torch.LongTensor(attention_mask), 0)
        return token_ids, attention_mask

