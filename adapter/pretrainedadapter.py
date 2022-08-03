import numpy as np
import pandas as pd
from config.data import DATASET_PATHS
from config.hyperparameters import ROBERTA_PARAMS
from datasets import ClassLabel
from datasets import load_dataset
from datasets import load_metric
from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding, AutoModelWithHeads, TextClassificationPipeline


def compute_metrics(eval_pred):
    metric = load_metric('accuracy')
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def evaluate(train_path, test_path, result_path, args):
    df = pd.read_json(train_path, lines=True)
    labels_df = df.label.unique()
    dataset = load_dataset("json", data_files={"train": train_path, "test": test_path})
    dataset = dataset.remove_columns(["metadata"])

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    labels = ClassLabel(names=labels_df)

    def preprocess_function(examples):
        tokenized = tokenizer(examples["text"], truncation=True, padding=True)
        tokenized['label'] = labels.str2int(examples['label'])
        return tokenized

    tokenized_citation = dataset.map(preprocess_function, batched=True)

    columns_to_return = ['input_ids', 'label', 'attention_mask']
    tokenized_citation.set_format(type='torch', columns=columns_to_return)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = AutoModelWithHeads.from_pretrained("roberta-base")
    adapter_name = model.load_adapter("AdapterHub/roberta-base-pf-scicite", source="hf")
    model.active_adapters = adapter_name
    training_args = TrainingArguments(output_dir=result_path,
                                      learning_rate=args["LEARNING_RATE"],
                                      per_device_train_batch_size=args["EVAL_BATCH_SIZE"],
                                      per_device_eval_batch_size=args["TRAIN_BATCH_SIZE"],
                                      num_train_epochs=args["NUM_EPOCHS"],
                                      weight_decay=args["WEIGHT_DECAY"])
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
    trainer.evaluate()
    classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer)
    print(classifier("These results are great for future purpose"))


if __name__ == '__main__':

    for key in DATASET_PATHS:
        print("Running classification task for", key, "dataset")
        evaluate(DATASET_PATHS[key]["train"], DATASET_PATHS[key]["test"], DATASET_PATHS[key]["result_base"], ROBERTA_PARAMS)
