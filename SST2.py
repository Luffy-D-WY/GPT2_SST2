from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import torch
if __name__=="__main__":

    # 检查GPU是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载数据集
    dataset = load_dataset("glue", "sst2")
    # glue为一组自然语言处理任务的集合 sst指明子任务

    # 初始化分词器
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    # 加载分词器，由于gpt2默认没有填充标记需将tokenizer填充标记设为结束标记

    # 预处理数据
    def preprocess_function(examples):
        return tokenizer(examples['sentence'], padding='max_length', truncation=True, max_length=128)
    # examples['sentence']返回了一个关于句子的列表
    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    # map方法接受一个函数作为参数
    # 加载预训练的GPT-2模型
    model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=2)
    model.resize_token_embeddings(len(tokenizer))  # 调整模型的词汇表大小以匹配分词器

    # 将模型移动到GPU
    model.to(device)

    # 设置模型的填充标记ID
    model.config.pad_token_id = tokenizer.pad_token_id

    # 设置训练参数
    training_args = TrainingArguments(
        output_dir='./results',
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,  # 使用较小的批次大小以适应显存
        per_device_eval_batch_size=16,
        num_train_epochs=4,  # 调整训练周期数
        weight_decay=0.01,
        fp16=True,  # 启用混合精度训练
        gradient_accumulation_steps=4,  # 启用梯度累积
        logging_steps=500,  # 减少日志记录的频率
        save_strategy="epoch",  # 每个epoch结束时保存模型
    )

    # 定义训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
    )

    # 开始训练
    trainer.train()

    # 评估模型
    results = trainer.evaluate()
    print(results)

    # 保存训练后的模型
    trainer.save_model("./trained_model3")
    tokenizer.save_pretrained("./trained_model3")
