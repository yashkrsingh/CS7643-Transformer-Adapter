
DATASETS = {
    "rct-20k": {
        "data_dir_train": "https://s3-us-west-2.amazonaws.com/allennlp/dont_stop_pretraining/data/rct-20k/train.jsonl",
        "data_dir_dev": "https://s3-us-west-2.amazonaws.com/allennlp/dont_stop_pretraining/data/rct-20k/dev.jsonl",
        "data_dir_test": "https://s3-us-west-2.amazonaws.com/allennlp/dont_stop_pretraining/data/rct-20k/test.jsonl",
        "dataset_size": 180040
    },
    "rct-sample": {
        "data_dir_train": "https://s3-us-west-2.amazonaws.com/allennlp/dont_stop_pretraining/data/rct-sample/train.jsonl",
        "data_dir_dev": "https://s3-us-west-2.amazonaws.com/allennlp/dont_stop_pretraining/data/rct-sample/dev.jsonl",
        "data_dir_test": "https://s3-us-west-2.amazonaws.com/allennlp/dont_stop_pretraining/data/rct-sample/test.jsonl",
        "dataset_size": 500
    },
    "citation_intent": {
        "data_dir_train": "https://s3-us-west-2.amazonaws.com/allennlp/dont_stop_pretraining/data/citation_intent/train.jsonl",
        "data_dir_dev": "https://s3-us-west-2.amazonaws.com/allennlp/dont_stop_pretraining/data/citation_intent/dev.jsonl",
        "data_dir_test": "https://s3-us-west-2.amazonaws.com/allennlp/dont_stop_pretraining/data/citation_intent/test.jsonl",
        "dataset_size": 1688
    }
}

DATASET_PATHS = {
    "citation_intent": {
        "train": "../data/citation_train.jsonl",
        "test": "../data/citation_test.jsonl",
        "result_base": "../results/citation_intent/base",
        "result_adapter": "../results/citation_intent/adapter"
    },

    "rct": {
        "train": "../data/rct20k_train.jsonl",
        "test": "../data/rct20k_test.jsonl",
        "result_base": "../results/rct20k/base",
        "result_adapter": "../results/rct20k/adapter"
    }
}
