import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

# 模型設定
model_checkpoint = 'distilbert-base-uncased'
id2label = {0: "Negative", 1: "Positive"}
label2id = {"Negative": 0, "Positive": 1}

# 建立模型
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint, 
    num_labels=2, 
    id2label=id2label, 
    label2id=label2id
)

# 建立 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# 資料集載入與處理
dataset = load_dataset("imdb")
def tokenize_function(examples):
    text = examples["text"]
    tokenizer.truncation_side = "left"
    tokenized_inputs = tokenizer(
        text,
        return_tensors="np",
        truncation=True,
        max_length=512
    )
    return tokenized_inputs

tokenized_dataset = dataset.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# LoRA 設定
peft_config = LoraConfig(
    task_type="SEQ_CLS",
    r=4,
    lora_alpha=32,
    lora_dropout=0.01,
    target_modules=['q_lin']
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# 訓練參數
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01
)

# 建立 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator
)

# 開始訓練
trainer.train()

# 確認裝置是否為 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 將模型移動到 GPU
model.to(device)

# 測試樣本列表
text_list = ["It was good.", "Not a fan, don't recommend.", "Better than the first one.", "This is not worth watching even once.", "This one is a pass."]

print("Untrained model predictions:")
print("----------------------------")
for text in text_list:
    # 將輸入移動到 GPU
    inputs = tokenizer.encode(text, return_tensors="pt").to(device)
    # 計算 logits
    logits = model(inputs).logits
    # 將 logits 轉換為標籤
    predictions = torch.argmax(logits)
    
    print(text + " - " + id2label[predictions.item()])
