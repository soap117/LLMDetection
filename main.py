# giving a prompt and response try to predict if the response is written by a human or a bot
import torch
import transformers
import csv
from typing import List
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
from datasets import load_dataset, Dataset
import evaluate
accuracy = evaluate.load("accuracy")
import numpy as np
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}
device_map = "cuda:0"

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

def load_model(model_path):
    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_path, cache_dir='/data/cache', torch_dtype=torch.bfloat16, num_labels=2, device_map=device_map, id2label=id2label, label2id=label2id)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, cache_dir='/data/cache')
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    return model, tokenizer

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=1024)

def apply_prompt(examples):
    prompt = examples[0]
    source_text = examples[1]
    response = examples[2]
    template = "Giving the following prompt: {} \n Source Text: {} \n Response: {}, is the response written by a human or a bot?"
    template_input = template.format(prompt, source_text, response)
    return template_input


if __name__ == '__main__':
    #load data from
    train_essays_file = '/home/jzl6599/kaggle_LLM/train_essays.csv'
    train_prompts_file = '/home/jzl6599/kaggle_LLM/train_prompts.csv'
    test_essays_file = '/home/jzl6599/kaggle_LLM/test_essays.csv'
    #loar config
    lora_r = 8
    lora_alpha = 16
    lora_dropout = 0.05
    lora_target_modules = ["q_proj", "v_proj", "k_proj"]

    #read the prompts and store in a dictionary
    prompts = {}
    with open(train_prompts_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        reader.__next__()
        for row in reader:
            #print(row)
            prompts[row[0]] = (row[1], row[2], row[3])
    print(prompts['0'])
    #read the essays and combine it with the prompts
    train_data = []
    with open(train_essays_file, 'r') as f:
        reader = csv.reader(f)
        reader.__next__()
        for row in reader:
            rid = row[0]
            prompt_id = row[1]
            response = row[2]
            label = row[3]
            prompt = prompts[prompt_id]
            text = apply_prompt((prompts[prompt_id][1], prompts[prompt_id][2], response))
            train_data.append({'text': text, 'label': int(label)})
    print(train_data[0])
    #read the test essays
    test_data = []
    model, tokenizer = load_model('meta-llama/Llama-2-7b-chat-hf')
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="SEQ_CLS",
        modules_to_save=['score'],
    )
    print(model)
    model = get_peft_model(model, config)

    model.print_trainable_parameters()
    #load data from the train_data
    test_data = train_data[-100:]
    train_data = train_data[:-100]
    train_dataset = Dataset.from_list(train_data)
    train_dataset = train_dataset.map(preprocess_function, batched=True)
    test_dataset = Dataset.from_list(test_data)
    test_dataset = test_dataset.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    training_args = TrainingArguments(
        output_dir="/data/junyu/kaggle_LLM",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    