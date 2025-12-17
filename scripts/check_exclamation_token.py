"""Check what token the exclamation mark corresponds to."""
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("allenai/dolma2-tokenizer", trust_remote_code=True)

# Check exclamation mark
exc_tokens = tokenizer.encode("!", add_special_tokens=False)
print(f"'!' tokenizes to: {exc_tokens}")

# Check if it's a special pattern
for i in range(30):
    print(f"Token {i}: '{tokenizer.decode([i])}'")

# Check what 30 exclamation marks decode from
thirty_exc = "!" * 30
tokens = tokenizer.encode(thirty_exc, add_special_tokens=False)
print(f"\n30 '!' characters tokenize to: {tokens}")
print(f"Number of tokens: {len(tokens)}")

# Check if there's a repeated token
if len(set(tokens)) == 1:
    print(f"All same token: {tokens[0]}")
    print(f"This token decodes to: '{tokenizer.decode([tokens[0]])}'")

