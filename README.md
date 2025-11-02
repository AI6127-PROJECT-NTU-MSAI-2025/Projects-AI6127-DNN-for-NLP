# 极简使用说明

#### 1.运行 [parser_notationed.py](parser_notationed.py) 生成./data/raw 下的原始.src,tgt文件 （代码只保留了中英文）

#### 2.运行 [preprocess_notationed.py](preprocess_notationed.py) 生成./data 下三种任务的.json文件

#### 3.给 [finetune_notationed.py](finetune_notationed.py) 设置运行参数

参数位置为
```python
if __name__ == "__main__":

    ########## 需要修改的参数 #########
    model_args = ModelArguments(
        model_name_or_path="google/mt5-small",  # 替换为您想使用的预训练模型路径, 如果输入huggingface模型名称下载的话，它会保存在
                                                # C:\Users\你的用户名\.cache\huggingface\hub
    )

    data_args = DataTrainingArguments(
        data_path="data/crosslingual",  # 数据位置
        data_name="En_Zh",  # 具体的任务
        train_file="train.jsonl",   #最终的训练集位置会被拼接为 data_path/data_name/train_file
        validation_file="dev.jsonl",
        max_source_length=512,
        max_target_length=128,
        val_max_target_length=64,
        num_beams=3,
        preprocessing_num_workers=4,
    )
    #不想设置的可以注释掉，会使用默认值
    training_args = Seq2SeqTrainingArguments(
        output_dir="./output_dir/mt5_finetune",  # !!! 替换为模型保存路径
        do_train=True,
        do_eval=True,
        num_train_epochs=1,
        max_steps=500,

        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,

        learning_rate=3e-5,  #初始学习率
        save_strategy="steps", #也可以用epoch
        save_steps=100,
        eval_strategy="steps",  #在部分 Seq2SeqTrainingArguments 版本中，这里可能需要修改为 evaluation_strategy
        eval_steps=100,
        load_best_model_at_end=True,  #评估表现最好的模型
        metric_for_best_model="bleu",  #评估依据  ,默认是loss
        greater_is_better=True,    #metric_for_best_model 是值越大越好 (如 BLEU) 还是越小越好 (如 Loss)
        predict_with_generate=True,
        early_stopping_patience=5,
        fp16=True,
        seed=42,
        report_to="none" # 若要启用日志记录器，请使用cmd运行，并输入你的API密钥，比如  wandb.login(key="[您的 API 密钥]")
    )
    ########## 需要修改的参数 (完) #########
```

#### 4.运行[finetune_notationed.py](finetune_notationed.py) 

##### 备注：Warning我没有管，会报的很热闹 
##### 备注:代码相较原始版本进行了一定修改，比如取消了原本dataset.py的使用。直接读json文件。代码的测试环境见requirements.txt，我没有为这个项目专门设置虚拟环境，所以有不少没用的，仅供包冲突时参考