from config.data import DATASET_PATHS
from config.hyperparameters import ROBERTA_TUNING
from transformer.robertabase import evaluate


if __name__ == '__main__':

    for key, item in ROBERTA_TUNING.items():
        print("Running model with params as", item)
        evaluate(DATASET_PATHS["citation_intent"]["train"],
                 DATASET_PATHS["citation_intent"]["test"],
                 DATASET_PATHS["citation_intent"]["result"],
                 item)
