import torch
from pathlib import Path
from tokenizers import Tokenizer

from train_config import get_config, latest_weights_file_path
from transformer.Transformer import build_transformer
from dataset import causal_mask

def load_tokenizer(config, lang):
    tokenizer_path = Path(config["tokenizer_file"].format(lang))
    if not tokenizer_path.exists():
        print(f"[ERROR] Tokenizer file not found: {tokenizer_path}")
        return None
    tok = Tokenizer.from_file(str(tokenizer_path))
    print(f"[OK] Loaded tokenizer for '{lang}' with vocab size: {tok.get_vocab_size()}")
    return tok

def greedy_decode_debug(model, source, source_mask, tokenizer_tgt, max_len, device):
    """Greedy decode with debugging output after each step."""
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")

    encoder_output = model.encode(source, source_mask)
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)

    step = 0
    while decoder_input.size(1) < max_len:
        step += 1
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)

        token_id = next_word.item()
        token_str = tokenizer_tgt.id_to_token(token_id)
        print(f"Step {step}: token_id={token_id} -> '{token_str}'")

        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(token_id).to(device)],
            dim=1
        )

        if token_id == eos_idx:
            print("[INFO] Reached EOS token, stopping decoding.")
            break

    return decoder_input.squeeze(0)

def main():
    config = get_config()
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if hasattr(torch, "has_mps") and torch.has_mps else "cpu"
    )
    print(f"[INFO] Using device: {device}")

    # Load tokenizers
    tokenizer_src = load_tokenizer(config, config["lang_src"])
    tokenizer_tgt = load_tokenizer(config, config["lang_tgt"])
    if not tokenizer_src or not tokenizer_tgt:
        return

    # Check presence of special tokens
    for tok, name in [(tokenizer_src, "source"), (tokenizer_tgt, "target")]:
        for special in ["[SOS]", "[EOS]", "[PAD]"]:
            tok_id = tok.token_to_id(special)
            if tok_id is None:
                print(f"[ERROR] Missing {special} in {name} tokenizer!")
            else:
                print(f"[OK] {name} {special} token id: {tok_id}")

    # Build model
    model = build_transformer(
        tokenizer_src.get_vocab_size(),
        tokenizer_tgt.get_vocab_size(),
        config["seq_len"],
        config["seq_len"],
        dmodel=config["d_model"]
    ).to(device)

    # Load weights
    model_filename = latest_weights_file_path(config)
    if not model_filename:
        print("[ERROR] No trained model weights found!")
        return
    print(f"[INFO] Loading model weights from: {model_filename}")
    state = torch.load(model_filename, map_location=device)

    missing, unexpected = model.load_state_dict(state["model_state_dict"], strict=False)
    if missing:
        print(f"[WARN] Missing keys in checkpoint: {missing}")
    if unexpected:
        print(f"[WARN] Unexpected keys in checkpoint: {unexpected}")
    print("[OK] Model weights loaded.")

    model.eval()

    # Run interactive debug translation
    while True:
        try:
            text = input("\nEnter a sentence in source language (or press Ctrl+C to quit): ")
            if not text.strip():
                continue

            enc = tokenizer_src.encode(text)
            tokens = enc.ids
            sos_idx = tokenizer_src.token_to_id("[SOS]")
            eos_idx = tokenizer_src.token_to_id("[EOS]")
            pad_idx = tokenizer_src.token_to_id("[PAD]")

            seq = [sos_idx] + tokens + [eos_idx]
            pad_len = config["seq_len"] - len(seq)
            if pad_len < 0:
                print("[WARN] Input sentence too long after tokenization!")
                continue
            seq += [pad_idx] * pad_len

            src_tensor = torch.tensor([seq], dtype=torch.int64).to(device)
            src_mask = (src_tensor != pad_idx).unsqueeze(0).unsqueeze(0).int()

            output_tokens = greedy_decode_debug(model, src_tensor, src_mask, tokenizer_tgt, config["seq_len"], device)

            out_ids = [i for i in output_tokens.tolist() if i not in [sos_idx, eos_idx, pad_idx]]
            decoded_text = tokenizer_tgt.decode(out_ids)

            print(f"[INFO] Raw output IDs: {output_tokens.tolist()}")
            print(f"[RESULT] Decoded translation: '{decoded_text}'")

        except KeyboardInterrupt:
            print("\nExiting debug session.")
            break

if __name__ == "__main__":
    main()
