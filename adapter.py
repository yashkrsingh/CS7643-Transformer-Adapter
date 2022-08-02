from datasets import load_dataset
import numpy as np
from datasets import load_metric
import pandas as pd
from transformers import RobertaConfig, RobertaAdapterModel, RobertaTokenizer
from transformers import TrainingArguments, AdapterTrainer, EvalPrediction
from transformers import TextClassificationPipeline


def main():
    df = pd.read_json(r'data\citation.jsonl', lines=True)
    labels_df = df.label.unique()
    dataset = load_dataset("json", data_files={"train": "data\citation.jsonl", "test": "data\citation_test.jsonl"})
    dataset = dataset.remove_columns(["metadata"])
    dataset['train'] = dataset['train'].class_encode_column("label")
    dataset['test'] = dataset['test'].class_encode_column("label")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    def encode_batch(batch):
        return tokenizer(batch["text"], max_length=80, truncation=True, padding="max_length")

    dataset = dataset.map(encode_batch, batched=True)
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    config = RobertaConfig.from_pretrained(
        "roberta-base",
        num_labels=labels_df.size,
    )
    model = RobertaAdapterModel.from_pretrained(
        "roberta-base",
        config=config,
    )
    # Add a new adapter
    model.add_adapter("roberta_base_rct")
    model.add_classification_head(
        "roberta_base_rct",
        num_labels=labels_df.size,
        id2label={0: labels_df[0], 1: labels_df[1], 2: labels_df[2], 3: labels_df[3], 4: labels_df[4]}
    )
    # Activate the adapter
    model.train_adapter("roberta_base_rct")
    training_args = TrainingArguments(
        learning_rate=1e-4,
        num_train_epochs=6,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        logging_steps=200,
        output_dir="./training_output",
        overwrite_output_dir=True,
        # The next line is important to ensure the dataset labels are properly passed to the model
        remove_unused_columns=False,
    )

    def compute_accuracy(p: EvalPrediction):
        preds = np.argmax(p.predictions, axis=1)
        return {"acc": (preds == p.label_ids).mean()}

    trainer = AdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        compute_metrics=compute_accuracy,
    )
    trainer.train()
    trainer.evaluate()
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer, device=training_args.device.index)
    print(classifier("These results are great for future purpose"))


if __name__ == '__main__':
    main()
