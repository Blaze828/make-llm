"""Byte-level BPE artifact with ordinary-text special-token isolation."""
import hashlib
import json
from pathlib import Path
import tokenizers
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

SPECIAL_TOKENS = ["<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>",
                  "<|system|>", "<|user|>", "<|assistant|>", "<|end_turn|>"]


class BPETokenizer:
    def __init__(self, backend, metadata=None):
        self.backend = backend
        self.metadata = metadata or {}
        # Treat control-token strings in user text as ordinary text.
        self.backend.encode_special_tokens = True

    @classmethod
    def train(cls, texts, vocab_size=32000, splitter=None, special_tokens=None):
        special_tokens = SPECIAL_TOKENS if special_tokens is None else special_tokens
        if len(set(special_tokens)) != len(special_tokens) or vocab_size < 256+len(special_tokens):
            raise ValueError("Vocabulary must contain unique special tokens and all 256 bytes")
        backend = Tokenizer(models.BPE())
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
        backend.decoder = decoders.ByteLevel()
        digest = hashlib.sha256()
        count = 0
        def pieces():
            nonlocal count
            for text in texts:
                if not isinstance(text, str):
                    raise TypeError("Tokenizer corpus must contain strings")
                raw = text.encode("utf-8")
                digest.update(len(raw).to_bytes(8, "little")); digest.update(raw)
                count += 1
                chunks = splitter(text) if splitter else [text]
                if "".join(chunks) != text:
                    raise ValueError("Morphology splitter changed source text")
                # Separate iterator items prevent merges across morphology boundaries.
                yield from (chunk for chunk in chunks if chunk)
        trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=1, show_progress=False,
                                      special_tokens=special_tokens,
                                      initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
        backend.train_from_iterator(pieces(), trainer=trainer)
        if not count:
            raise ValueError("Empty tokenizer corpus")
        metadata = {"format_version": 1, "family": "morph_byte_bpe" if splitter else "byte_bpe",
                    "requested_vocab_size": vocab_size, "actual_vocab_size": backend.get_vocab_size(),
                    "corpus_sha256": digest.hexdigest(), "document_count": count,
                    "tokenizers_version": tokenizers.__version__, "special_tokens": special_tokens,
                    "morphology_scope": "vocabulary_training_only" if splitter else "none"}
        if splitter and hasattr(splitter, "metadata"):
            metadata["morphology"] = splitter.metadata
        return cls(backend, metadata)

    @property
    def vocab_size(self):
        return self.backend.get_vocab_size()

    def special_id(self, token):
        if token not in self.metadata.get("special_tokens", SPECIAL_TOKENS):
            raise ValueError("Not a registered control token")
        return self.backend.token_to_id(token)

    def encode(self, text):
        return self.backend.encode(text, add_special_tokens=False).ids

    def decode(self, ids):
        return self.backend.decode(ids, skip_special_tokens=False)

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.backend.save(str(path / "tokenizer.json"))
        metadata = dict(self.metadata)
        metadata["tokenizer_sha256"] = hashlib.sha256((path / "tokenizer.json").read_bytes()).hexdigest()
        (path / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        path = Path(path)
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        digest = hashlib.sha256((path / "tokenizer.json").read_bytes()).hexdigest()
        if digest != metadata["tokenizer_sha256"]:
            raise ValueError("Tokenizer artifact hash mismatch")
        return cls(Tokenizer.from_file(str(path / "tokenizer.json")), metadata)
