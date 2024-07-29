from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW

class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

    def __len__(self):
        return len(self.texts)

def get_dataloader(dataset, tokenizer, batch_size):
    train_texts = dataset['train']['sentence']
    train_labels = dataset['train']['label']
    val_texts = dataset['validation']['sentence']
    val_labels = dataset['validation']['label']
    test_texts = dataset['test']['sentence']
    test_labels = dataset['test']['label']

    train_dataset = CustomDataset(train_texts, train_labels, tokenizer)
    val_dataset = CustomDataset(val_texts, val_labels, tokenizer)
    test_dataset = CustomDataset(test_texts, test_labels, tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    return train_dataloader, val_dataloader, test_dataloader

def train(model, train_dataloader, learning_rate, gradient_accumulation_steps, device):
    #使用model和dataploader训练一个epoch。一个epoch的意思就是训练集里的所有sample都参与训练一次
    model.train()
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_loss = 0

    for step, batch in enumerate(train_dataloader):
        #循环每个batch, 将token id输进model里获得output，并根据output以及label得到loss，并得到梯度
        #每gradient_accumulation_steps个batch后，进行一次梯度下降
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss / gradient_accumulation_steps
        loss.backward()

        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_dataloader)
    print(f"Training loss: {avg_loss}")
    #在这一整个epoch的过程中，还要跟踪平均loss等指标，以记录训练过程

def eval(model, dataloader, device):
    #使用验证集验证当前model
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            total_loss += loss.item()

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_predictions
    print(f"Evaluation loss: {avg_loss}, Accuracy: {accuracy}")
    return avg_loss, accuracy
def save(model, path="./best_model"):
    model.save_pretrained(path)

def load(path="./best_model", device='cpu'):
    model = GPT2ForSequenceClassification.from_pretrained(path)
    model.to(device)
    return model

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=2)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    model.config.pad_token_id = tokenizer.pad_token_id

    dataset = load_dataset("glue", "sst2")
    batch_size = 16
    train_dataloader, val_dataloader, test_dataloader = get_dataloader(dataset, tokenizer, batch_size)

    num_train_epochs = 5
    learning_rate = 2e-5
    gradient_accumulation_steps = 4

    best_loss = float('inf')
    best_epoch = 0
    #每个epoch，在训练集里所有样本上训练一次，并使用验证集计算当前训练质量
    #还要把最好的一个epoch对应的模型给保存下来，以作为整个训练过程的结果。因为我们最后得到的model不是训练完4个epoch后的model，而是4个epoch里最好的那个epoch。
    #评价哪个epoch对应model“最好”的标准，是看哪个epoch的model在验证集上的平均loss最低，所以才需要在每个epoch调用eval()函数来计算当前epoch经过train之后在验证集上的loss
    for epoch in range(num_train_epochs):
        print(f"Epoch {epoch+1}/{num_train_epochs}")
        train(model, train_dataloader, learning_rate, gradient_accumulation_steps, device)
        eval_loss, eval_accuracy = eval(model, val_dataloader, device)

        if eval_loss < best_loss:
            best_loss = eval_loss
            best_epoch = epoch
            save(model, path=f"./best_model_epoch_{epoch}")

    print(f"Best epoch: {best_epoch+1}")
    model = load(f"./best_model_epoch_{best_epoch}", device=device)
    #在根据eval数据集挑选出最好的epoch之后，还要使用test数据集计算出准确率和loss作为最终结果
    #print("Testing the best model on the test set:")
    #eval(model, test_dataloader, device)
