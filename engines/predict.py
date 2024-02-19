# -*- coding: utf-8 -*-
# @Author : Stanley
# @Email : gzlishouxian@gmail.com
# @File : predict.py
# @Software: VSCode
import torch
import os
import time
import json
import numpy as np
from torch.utils.data import DataLoader
from engines.model import NLUModel
from onnxruntime import InferenceSession, SessionOptions


class Predictor:
    def __init__(self, configs, data_manager, device, logger):
        self.device = device
        self.configs = configs
        self.data_manager = data_manager
        self.logger = logger
        self.predict_engine = configs['predict_engine']
        self.checkpoints_dir = configs['checkpoints_dir']
        self.max_sequence_length = data_manager.max_sequence_length
        if self.predict_engine == 'onnx':
            if str(device) == 'cuda':
                providers = ['CUDAExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']
            sess_options = SessionOptions()
            onnx_model = os.path.join(self.checkpoints_dir, 'model.onnx')
            self.model = InferenceSession(onnx_model, sess_options=sess_options, providers=providers)
            if device == 'cuda':
                try:
                    assert 'CUDAExecutionProvider' in self.model.get_providers()
                except AssertionError:
                    raise AssertionError(
                        'The environment for GPU inference is not set properly. '
                        'A possible cause is that you had installed both onnxruntime and onnxruntime-gpu. '
                        'Please run the following commands to reinstall: \n '
                        '1) pip uninstall -y onnxruntime onnxruntime-gpu \n 2) pip install onnxruntime-gpu'
                    )
            self.logger.info('>>> [InferBackend] Engine Created ...')
        else:
            self.model_name = configs['model_name']
            self.model = NLUModel(
                self.data_manager.entity_num_labels, self.data_manager.intent_num_labels, self.device).to(self.device)
            self.model.load_state_dict(torch.load(os.path.join(self.checkpoints_dir, self.model_name)))
            self.model.eval()

    def predict_test(self):
        test_file = self.configs['test_file']
        if test_file == '':
            self.logger.error('test dataset not found...')
            raise Exception('test dataset not found...')
        from engines.train import Train
        train = Train(self.configs, self.data_manager, self.device, self.logger)
        self.logger.info('loading test data...')
        test_data = json.load(open(test_file, encoding='utf-8'))
        test_loader = DataLoader(
            dataset=test_data,
            batch_size=train.batch_size,
            collate_fn=self.data_manager.prepare_data
        )
        self.logger.info('test dataset nums:{}'.format(len(test_loader)))
        train.validate(self.model, test_loader)

    def predict_one(self, sentence):
        start_time = time.time()
        token_ids, attention_mask = self.data_manager.prepare_single_sentence(sentence)

        if self.predict_engine == 'onnx':
            token_ids = np.array(token_ids.tolist(), dtype=np.int32)
            infer_input = {'tokens': token_ids}
            entity_logits, _, intent_logits, intent_pro = self.model.run(None, dict(infer_input))
            entity_logits = torch.Tensor(np.squeeze(entity_logits))
            intent_logits = torch.Tensor(intent_logits)
            intent_pro = torch.Tensor(intent_pro)
        else:
            token_ids = token_ids.to(self.device)
            entity_logits, _, intent_logits, intent_pro = self.model(token_ids)
            entity_logits = torch.squeeze(entity_logits.to('cpu'))
        entity_results = self.data_manager.extract_entities(sentence, entity_logits, inference=True)
        entity_dict = {}
        for class_id, result_set in entity_results.items():
            entity_dict[self.data_manager.entity_reverse_categories[class_id]] = list(result_set)
        intent_prediction = torch.argmax(intent_logits.to('cpu'), dim=-1).numpy()[0]
        this_probability = list(intent_pro.tolist()[0])[intent_prediction]
        self.logger.info('predict time consumption: %.3f(ms)' % ((time.time() - start_time) * 1000))
        return entity_dict, self.data_manager.intent_reverse_classes[str(intent_prediction)], this_probability

    def convert_onnx(self):
        dummy_input = torch.ones([1, 1]).to('cpu').int()
        onnx_path = self.checkpoints_dir + '/model.onnx'
        torch.onnx.export(self.model.to('cpu'), dummy_input, f=onnx_path, opset_version=14,
                          input_names=['tokens'], output_names=['slot_logits', 'slot_probs', 'intent_logits', 'intent_probs'],
                          do_constant_folding=True,
                          dynamic_axes={'tokens': {0: 'batch_size', 1: 'sequence_lens'},
                                        'slot_logits': {0: 'batch_size'},
                                        'slot_probs': {0: 'batch_size'},
                                        'intent_logits': {0: 'batch_size'},
                                        'intent_probs': {0: 'batch_size'}})
        self.logger.info('convert torch to onnx successful...')
