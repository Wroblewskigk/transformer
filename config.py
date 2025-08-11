from pathlib import Path


# noinspection SpellCheckingInspection
def get_config():
    return {
        "batch_size": 16,
        "num_epochs": 1,
        "lr": 1e-4,
        "seq_len": 25,
        "d_model": 512,
        "datasource": "Helsinki-NLP/opus-100",
        "lang_src": "en",
        "lang_tgt": "pl",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "latest",
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel",
    }


def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path(".") / model_folder / model_filename)


def latest_weights_file_path(config):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])
