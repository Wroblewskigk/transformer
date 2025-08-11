import torch
from model import build_transformer
from config import get_config

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build the model with config vocab sizes and sequence length
    # Assuming you have tokenizers or you know vocab sizes, for example 30000 here
    src_vocab_size = 30000
    tgt_vocab_size = 30000
    model = build_transformer(
        src_vocab_size,
        tgt_vocab_size,
        config["seq_len"],
        config["seq_len"],
        dmodel=config["d_model"],
    ).to(device)

    print("\nModel Architecture:\n")
    print(model)

    total_params = count_parameters(model)
    print(f"\nTotal trainable parameters: {total_params:,}")

if __name__ == "__main__":
    main()
