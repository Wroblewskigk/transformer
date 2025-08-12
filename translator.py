from pathlib import Path

import torch
from tokenizers import Tokenizer

from train_config import get_config, latest_weights_file_path
from dataset import causal_mask
from transformer.Transformer import build_transformer


def greedy_decode(model, source, source_mask, tokenizer_tgt, max_len, device):

    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")
    encoder_output = model.encode(source, source_mask)
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)

    while decoder_input.size(1) < max_len:
        decoder_mask = (
            causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        )
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [
                decoder_input,
                torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device),
            ],
            dim=1,
        )
        if next_word.item() == eos_idx:
            break
    return decoder_input.squeeze(0)


def load_tokenizer(config, lang):

    tokenizer_path = Path(config["tokenizer_file"].format(lang))
    return Tokenizer.from_file(str(tokenizer_path))


def main():

    config = get_config()
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if hasattr(torch, "has_mps") and torch.has_mps else "cpu"
    )

    # Load Tokenizers
    tokenizer_src = load_tokenizer(config, config["lang_src"])
    tokenizer_tgt = load_tokenizer(config, config["lang_tgt"])

    # Build Model
    model = build_transformer(
        tokenizer_src.get_vocab_size(),
        tokenizer_tgt.get_vocab_size(),
        config["seq_len"],
        config["seq_len"],
        dmodel=config["d_model"],
    ).to(device)

    # Load Weights
    model_filename = latest_weights_file_path(config)
    if not model_filename:
        print("No trained model weights found!")
        return
    state = torch.load(model_filename, map_location=device)
    model.load_state_dict(state["model_state_dict"])

    model.eval()
    print("Type a sentence in English (or press Ctrl+C to quit):")
    while True:
        try:
            text = input("> ")
            if not text.strip():
                continue

            # Tokenize input
            encoded = tokenizer_src.encode(text)
            tokens = encoded.ids

            # Pad and add special tokens
            sos_idx = tokenizer_src.token_to_id("[SOS]")
            eos_idx = tokenizer_src.token_to_id("[EOS]")
            pad_idx = tokenizer_src.token_to_id("[PAD]")
            seq = [sos_idx] + tokens + [eos_idx]
            pad_length = config["seq_len"] - len(seq)
            if pad_length < 0:
                print("Input sentence too long!")
                continue
            seq += [pad_idx] * pad_length
            src_tensor = torch.tensor([seq], dtype=torch.int64).to(device)
            src_mask = (src_tensor != pad_idx).unsqueeze(0).unsqueeze(0).int()

            # Inference
            output_tokens = greedy_decode(
                model, src_tensor, src_mask, tokenizer_tgt, config["seq_len"], device
            )

            # Remove special tokens
            out_ids = output_tokens.tolist()
            out_text = tokenizer_tgt.decode(
                [
                    i
                    for i in out_ids
                    if i
                    not in [
                        tokenizer_tgt.token_to_id("[SOS]"),
                        tokenizer_tgt.token_to_id("[EOS]"),
                        tokenizer_tgt.token_to_id("[PAD]"),
                    ]
                ]
            )
            print("Translation:", out_text)
        except KeyboardInterrupt:
            print("\nExiting...")
            break


if __name__ == "__main__":
    main()
