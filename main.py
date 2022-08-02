from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
import numpy as np
from datasets import load_metric
from datasets import ClassLabel
import pandas as pd



def compute_metrics(eval_pred):
    metric = load_metric('accuracy')
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def main():
    df = pd.read_json(r'data\citation.jsonl', lines=True)
    labels_df = df.label.unique()
    dataset = load_dataset("json", data_files={"train": "data\citation.jsonl", "test": "data\citation_test.jsonl"})
    dataset = dataset.remove_columns(["metadata"])
    print(dataset)
    print("step 1")
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    labels = ClassLabel(names=labels_df)

    def preprocess_function(examples):
        tokenized = tokenizer(examples["text"], truncation=True, padding=True)
        tokenized['label'] = labels.str2int(examples['label'])
        return tokenized

    print("step 2")
    tokenized_citation = dataset.map(preprocess_function, batched=True)
    print(tokenized_citation)
    print("step 3")
    columns_to_return = ['input_ids', 'label', 'attention_mask']
    tokenized_citation.set_format(type='torch', columns=columns_to_return)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=labels_df.size)
    training_args = TrainingArguments(output_dir="./results", learning_rate=2e-5, per_device_train_batch_size=16,
                                      per_device_eval_batch_size=16, num_train_epochs=5, weight_decay=0.01, )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_citation["train"],
        eval_dataset=tokenized_citation["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    trainer.train()


if __name__ == '__main__':
    main()
