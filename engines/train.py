# -*- coding: utf-8 -*-
# @Author : Stanley
# @Email : gzlishouxian@gmail.com
# @File : train.py
# @Software: VSCode
from tqdm import tqdm
from torch.utils.data import DataLoader
from engines.utils.metrics import cal_metrics
import numpy as np
import json
import torch
import time
import os


class Train:
    def __init__(self, configs, data_manager, device, logger):
        self.configs = configs
        self.device = device
        self.logger = logger
        self.data_manager = data_manager
        self.optimizer = None
        self.learning_rate = configs['learning_rate']
        self.batch_size = configs['batch_size']
        self.checkpoints_dir = configs['checkpoints_dir']
        self.model_name = configs['model_name']
        self.epoch = configs['epoch']
        self.slot_model = configs['slot_model']
        if configs['use_focal_loss']:
            from engines.utils.losses.focal_loss import FocalLoss
            self.intent_loss = FocalLoss(device)
        else:
            self.intent_loss = torch.nn.CrossEntropyLoss()
        if configs['use_multilabel_categorical_cross_entropy']:
            from engines.utils.losses.multilabel_ce import MultilabelCategoricalCrossEntropy
            self.entity_loss = MultilabelCategoricalCrossEntropy()
        else:
            self.entity_loss = torch.nn.BCEWithLogitsLoss(reduction='none')

    def init_model(self):
        from engines.model import NLUModel
        model = NLUModel(
            self.data_manager.entity_num_labels, self.data_manager.intent_num_labels, self.device).to(self.device)
        optimizer_type = self.configs['optimizer']
        no_decay = ['bias', 'LayerNorm.weight']
        params = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay':  0.0},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        if optimizer_type == 'Adagrad':
            self.optimizer = torch.optim.Adagrad(params, lr=self.learning_rate)
        elif optimizer_type == 'Adadelta':
            self.optimizer = torch.optim.Adadelta(params, lr=self.learning_rate)
        elif optimizer_type == 'RMSprop':
            self.optimizer = torch.optim.RMSprop(params, lr=self.learning_rate)
        elif optimizer_type == 'SGD':
            self.optimizer = torch.optim.SGD(params, lr=self.learning_rate)
        elif optimizer_type == 'Adam':
            self.optimizer = torch.optim.Adam(params, lr=self.learning_rate)
        elif optimizer_type == 'AdamW':
            self.optimizer = torch.optim.AdamW(params, lr=self.learning_rate)
        else:
            raise Exception('optimizer_type does not exist')
        return model

    def split_data(self):
        train_file = self.configs['train_file']
        dev_file = self.configs['dev_file']
        train_data, dev_data = None, None
        train_data = json.load(open(train_file, encoding='utf-8'))
        if dev_file != '':
            dev_data = json.load(open(dev_file, encoding='utf-8'))

        if dev_file == '':
            self.logger.info('generate validation dataset...')
            validation_rate = self.configs['validation_rate']
            ratio = 1 - validation_rate
            train_data, dev_data = train_data[:int(ratio * len(train_data))], train_data[int(ratio * len(train_data)):]

        self.logger.info('loading train data...')
        train_loader = DataLoader(
            dataset=train_data,
            batch_size=self.batch_size,
            collate_fn=self.data_manager.prepare_data,
            shuffle=True
        )
        self.logger.info('loading validation data...')
        dev_loader = DataLoader(
            dataset=dev_data,
            batch_size=self.batch_size,
            collate_fn=self.data_manager.prepare_data
        )
        self.logger.info('train dataset nums:{}'.format(len(train_data)))
        self.logger.info('validation dataset nums:{}'.format(len(dev_data)))
        return train_loader, dev_loader

    def train(self):
        model = self.init_model()
        if os.path.exists(os.path.join(self.checkpoints_dir, self.model_name)):
            self.logger.info('Resuming from checkpoint...')
            model.load_state_dict(torch.load(os.path.join(self.checkpoints_dir, self.model_name)))
            optimizer_checkpoint = torch.load(os.path.join(self.checkpoints_dir, self.model_name + '.optimizer'))
            self.optimizer.load_state_dict(optimizer_checkpoint['optimizer'])
        else:
            self.logger.info('Initializing from scratch.')

        train_loader, dev_loader = self.split_data()
        self.train_each_fold(model, train_loader, dev_loader)

    def train_each_fold(self, model, train_loader, dev_loader):
        best_f1 = 0
        best_epoch = 0
        unprocessed = 0
        step_total = self.epoch * len(train_loader)
        global_step = 0
        scheduler = None

        if self.configs['warmup']:
            scheduler_type = self.configs['scheduler_type']
            if self.configs['num_warmup_steps'] == -1:
                num_warmup_steps = step_total * 0.1
            else:
                num_warmup_steps = self.configs['num_warmup_steps']

            if scheduler_type == 'linear':
                from transformers.optimization import get_linear_schedule_with_warmup
                scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                            num_warmup_steps=num_warmup_steps,
                                                            num_training_steps=step_total)
            elif scheduler_type == 'cosine':
                from transformers.optimization import get_cosine_schedule_with_warmup
                scheduler = get_cosine_schedule_with_warmup(optimizer=self.optimizer,
                                                            num_warmup_steps=num_warmup_steps,
                                                            num_training_steps=step_total)
            else:
                raise Exception('scheduler_type does not exist')

        very_start_time = time.time()
        for i in range(self.epoch):
            self.logger.info('\nepoch:{}/{}'.format(i + 1, self.epoch))
            model.train()
            start_time = time.time()
            step, loss, loss_sum = 0, 0.0, 0.0
            for batch in tqdm(train_loader):
                _, token_ids, attention_mask, intent_labels, _, entity_vectors = batch
                token_ids = token_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                entity_vectors = entity_vectors.to(self.device)
                intent_labels = intent_labels.to(self.device)
                self.optimizer.zero_grad()
                entity_logits, _, intent_logits, _ = model(token_ids)
                batch_size = entity_logits.size(0)
                if self.configs['use_multilabel_categorical_cross_entropy']:
                    if self.slot_model == 'bp':
                        num_labels = self.data_manager.entity_num_labels * 2
                    else:
                        num_labels = self.data_manager.entity_num_labels
                    model_output = entity_logits.reshape(batch_size * num_labels, -1)
                    label_vectors = entity_vectors.reshape(batch_size * num_labels, -1)
                    entity_loss = self.entity_loss(model_output, label_vectors)
                else:
                    if self.slot_model == 'bp':
                        entity_loss = self.entity_loss(entity_logits, entity_vectors)
                        entity_loss = torch.sum(torch.mean(entity_loss, 3), 2)
                        entity_loss = torch.sum(entity_loss * attention_mask) / torch.sum(attention_mask)
                    else:
                        model_output = entity_logits.reshape(batch_size * self.data_manager.entity_num_labels, -1)
                        label_vectors = entity_logits.reshape(batch_size * self.data_manager.entity_num_labels, -1)
                        entity_loss = self.entity_loss(model_output, label_vectors).mean()

                intent_loss = self.intent_loss(intent_logits, intent_labels)
                loss = 2 * entity_loss + intent_loss
                loss.backward()
                loss_sum += loss.item()
                self.optimizer.step()

                if self.configs['warmup']:
                    scheduler.step()

                if step % self.configs['print_per_batch'] == 0 and step != 0:
                    avg_loss = loss_sum / (step + 1)
                    self.logger.info('training_loss:%f' % avg_loss)

                step = step + 1
                global_step = global_step + 1

            f1 = self.validate(model, dev_loader)
            time_span = (time.time() - start_time) / 60
            self.logger.info('time consumption:%.2f(min)' % time_span)
            if f1 >= best_f1:
                unprocessed = 0
                best_f1 = f1
                best_epoch = i + 1
                optimizer_checkpoint = {'optimizer': self.optimizer.state_dict()}
                torch.save(optimizer_checkpoint, os.path.join(self.checkpoints_dir, self.model_name + '.optimizer'))
                torch.save(model.state_dict(), os.path.join(self.checkpoints_dir, self.model_name))
                self.logger.info('saved model successful...')
            else:
                unprocessed += 1
            aver_loss = loss_sum / step
            self.logger.info(
                'aver_loss: %.4f, f1: %.4f, best_f1: %.4f, best_epoch: %d, steps: %d \n' % (aver_loss, f1, best_f1,
                                                                                            best_epoch, global_step))
            if self.configs['is_early_stop']:
                if unprocessed > self.configs['patient']:
                    self.logger.info('early stopped, no progress obtained within {} epochs'.format(
                        self.configs['patient']))
                    self.logger.info('overall best f1 is {} at {} epoch'.format(best_f1, best_epoch))
                    self.logger.info('total training time consumption: %.3f(min)' % (
                            (time.time() - very_start_time) / 60))
                    return
        self.logger.info('overall best f1 is {} at {} epoch'.format(best_f1, best_epoch))
        self.logger.info('total training time consumption: %.3f(min)' % ((time.time() - very_start_time) / 60))

    def validate(self, model, dev_loader):
        counts = {}
        results_of_each_entity = {}
        for class_name, class_id in self.data_manager.entity_categories.items():
            counts[class_id] = {'A': 0.0, 'B': 1e-10, 'C': 1e-10}
            class_name = self.data_manager.entity_reverse_categories[class_id]
            results_of_each_entity[class_name] = {}

        with torch.no_grad():
            model.eval()
            y_true, y_pred = np.array([]), np.array([])
            self.logger.info('start evaluate engines...')
            for batch in tqdm(dev_loader):
                texts, token_ids, attention_mask, intent_labels, entity_results, _ = batch
                token_ids = token_ids.to(self.device)
                entity_logits, _, intent_logits, _ = model(token_ids)

                results = entity_logits.to('cpu')
                for text, result, entity_result in zip(texts, results, entity_results):
                    p_results = self.data_manager.extract_entities(text, result)
                    for class_id, entity_set in entity_result.items():
                        p_entity_set = p_results.get(class_id)
                        if p_entity_set is None:
                            # 没预测出来
                            p_entity_set = set()
                        # 预测出来并且正确个数
                        counts[class_id]['A'] += len(p_entity_set & entity_set)
                        # 预测出来的结果个数
                        counts[class_id]['B'] += len(p_entity_set)
                        # 真实的结果个数
                        counts[class_id]['C'] += len(entity_set)

                predictions = torch.argmax(intent_logits, dim=-1)
                y_true = np.append(y_true, intent_labels.to('cpu'))
                y_pred = np.append(y_pred, predictions.to('cpu'))

        for class_id, count in counts.items():
            f1, precision, recall = 2 * count['A'] / (
                    count['B'] + count['C']), count['A'] / count['B'], count['A'] / count['C']
            class_name = self.data_manager.entity_reverse_categories[class_id]
            results_of_each_entity[class_name]['f1'] = f1
            results_of_each_entity[class_name]['precision'] = precision
            results_of_each_entity[class_name]['recall'] = recall

        # 打印每一个实体识别的指标
        entity_f1 = 0.0
        for entity, performance in results_of_each_entity.items():
            entity_f1 += performance['f1']
            # 打印每个类别的指标
            self.logger.info('entity_name: %s, precision: %.4f, recall: %.4f, f1: %.4f'
                        % (entity, performance['precision'], performance['recall'], performance['f1']))
        # 这里算得是所有类别的平均f1值
        entity_f1 = entity_f1 / len(results_of_each_entity)

        measures, each_intent_classes = cal_metrics(y_true=y_true, y_pred=y_pred)

        # 打印每一个意图分类的指标
        classes_val_str = ''
        for k, v in each_intent_classes.items():
            try:
                map_k = str(int(float(k)))
            except ValueError:
                continue
            if map_k in self.data_manager.intent_reverse_classes:
                classes_val_str += (
                        self.data_manager.intent_reverse_classes[map_k] + ': ' + str(each_intent_classes[k]) + '\n')
        self.logger.info(classes_val_str)
        intent_f1 = measures['f1']
        f1 = (intent_f1 + entity_f1)/2
        return f1
