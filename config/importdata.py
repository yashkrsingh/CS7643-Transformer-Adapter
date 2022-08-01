from config.data import DATASETS
import requests


def saveDatasets(response, filename):
    with open(filename, "w") as f:
        f.write(response.text)

def getDatasets():
    citation_train = requests.get(DATASETS['citation_intent']['data_dir_train'])
    citation_test = requests.get(DATASETS['citation_intent']['data_dir_test'])

    rct20k_train = requests.get(DATASETS['rct-20k']['data_dir_train'])
    rct20k_test = requests.get(DATASETS['rct-20k']['data_dir_test'])

    saveDatasets(citation_train, "../data/citation_train.jsonl")
    saveDatasets(citation_test, "../data/citation_test.jsonl")
    saveDatasets(rct20k_train, "../data/rct20k_train.jsonl")
    saveDatasets(rct20k_test, "../data/rct20k_test.jsonl")


if __name__ == '__main__':
    getDatasets()