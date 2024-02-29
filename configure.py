# -*- coding: utf-8 -*-
# @Author : Stanley
# @Email : gzlishouxian@gmail.com
# @File : configure.py
# @Software: VSCode

# 模式
# train:                训练分类器
# interactive_predict:  交互模式
# test:                 跑测试集
# convert_onnx:         将torch模型保存onnx文件
# show_model_info:      打印模型参数
mode = 'interactive_predict'

# 使用GPU设备
use_cuda = True
cuda_device = 0


configure = {
    # 训练数据集
    'train_file': 'datasets/train.json',
    # 验证数据集
    'dev_file': '',
    'validation_rate': 0.1,
    # 测试数据集
    'test_file': 'datasets/train.json',
    # 使用的预训练模型
    'ptm': 'hfl/chinese-bert-wwm-ext',
    # 抽取模型方式
    'slot_model': 'bp',
    # 预测加载方式
    'predict_engine': 'pytorch',
    # 模型保存的文件夹
    'checkpoints_dir': 'checkpoints/Bert_2024021913',
    # 模型名字
    'model_name': 'nlu.pkl',
    'entity_classes': ['to', 'cc', 'bcc', 'subject', 'content', 'words', 'type', 'style', 'lang',
                       'event', 'event_time', 'notify_time', 'repeat_type', 'year', 'month', 'week', 'date', 'hour', 'time'],
    'intent_classes': ['eml_w', 'eml_re', 'eml_abs', 'eml_continue', 'eml_polish', 'eml_trans', 'eml_correct', 'eml_change_style', 'eml_change_words',
                       'op_ask_ability', 'op_cancel', 'op_send', 'op_reply', 'op_reply_all', 'op_insert', 'op_replace', 'op_todo_add', 'op_confirm', 'op_skip', 'op_retry', 'op_supply_slot',
                       'todo_sum', 'todo_query', 'todo_add', 'todo_add_from_mail',
                       'sec_danger_command', 'sec_identity',
                       'customer_support', 'others', 'no_intent'],
    # 随机种子
    'seed': 3407,
    'use_multilabel_categorical_cross_entropy': True,
    # 类别样本比例失衡的时候可以考虑使用
    'use_focal_loss': True,
    # focal loss的各个标签权重
    'weight': None,
    # 是否进行warmup
    'warmup': True,
    # warmup方法，可选：linear、cosine
    'scheduler_type': 'linear',
    # warmup步数，-1自动推断为总步数的0.1
    'num_warmup_steps': -1,
    # 句子最大长度
    'max_sequence_length': 300,
    # decision_threshold
    'decision_threshold': 0.5,
    # epoch
    'epoch': 60,
    # batch_size
    'batch_size': 64,
    # dropout rate
    'dropout_rate': 0.5,
    # 多分类使用micro或macro
    'metrics_average': 'micro',
    # 每print_per_batch打印损失函数
    'print_per_batch': 200,
    # learning_rate
    'learning_rate': 5e-5,
    # 优化器选择
    'optimizer': 'AdamW',
    # 执行权重初始化，仅限于非微调
    'init_network': False,
    # 训练是否提前结束微调
    'is_early_stop': False,
    # 训练阶段的patient
    'patient': 10,
}
