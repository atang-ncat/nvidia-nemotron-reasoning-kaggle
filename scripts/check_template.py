#!/usr/bin/env python3
"""Check the Nemotron base model's tokenizer for chat template info."""
from transformers import AutoTokenizer

MODEL = "/scratch2/atang/competitions/nemotron-kaggle/models/nemotron-base"
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

# Check known special tokens
test_tokens = [
    "\x3c|im_start|\x3e",
    "\x3c|im_end|\x3e",
    "\x3cextra_id_0\x3e",
    "\x3cextra_id_1\x3e",
    "\x3c|user|\x3e",
    "\x3c|assistant|\x3e",
    "\x3c|system|\x3e",
]

print("Token existence check:")
for t in test_tokens:
    ids = tok.encode(t, add_special_tokens=False)
    # If it encodes to a single token, it's a special token
    if len(ids) == 1:
        print(f"  {t!r} -> single token ID {ids[0]} (SPECIAL)")
    else:
        print(f"  {t!r} -> {len(ids)} tokens: {ids[:5]} (not special)")

# Check what format the Kaggle submission notebook uses
# Usually the competition provides a reference notebook
print()
print("EOS token:", repr(tok.eos_token), "->", tok.eos_token_id)
print("BOS token:", repr(tok.bos_token), "->", tok.bos_token_id)

# Try ChatML format
print()
chatml_prompt = "\x3c|im_start|\x3euser\nWhat is 2+2?\x3c|im_end|\x3e\n\x3c|im_start|\x3eassistant\n"
ids = tok.encode(chatml_prompt, add_special_tokens=False)
print(f"ChatML encoding ({len(ids)} tokens): {ids[:10]}...")
decoded = tok.decode(ids)
print(f"Decoded: {decoded[:100]!r}")

# Compare with our current format
print()
our_prompt = "### Question:\nWhat is 2+2?\n\n### Answer:\n"
ids2 = tok.encode(our_prompt, add_special_tokens=False)
print(f"Our format encoding ({len(ids2)} tokens): {ids2[:10]}...")
