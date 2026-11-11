"""Train-only structural vocabulary plus a fixed, universal byte alphabet."""

from dataclasses import dataclass

from .structured_codec import BYTE_TOKENS, FORMAT, POINTER_TOKENS


VOCAB_FORMAT = "typed_action_vocabulary_v1"
RESERVED = ("<PAD>", "<BOS>", "<UNK>")


class VocabularyError(ValueError):
    pass


@dataclass(frozen=True)
class StructuredVocabulary:
    tokens: tuple[str, ...]

    def __post_init__(self):
        base = (*RESERVED, *BYTE_TOKENS)
        if (self.tokens[:len(base)] != base or len(set(self.tokens)) != len(self.tokens)
                or any(not isinstance(token, str) or not token for token in self.tokens)):
            raise VocabularyError("invalid typed-action vocabulary")

    @classmethod
    def fit(cls, operation_sequences, *, split="train"):
        if split != "train":
            raise VocabularyError("structural vocabulary may only fit train records")
        observed = {operation["token"] for operations in operation_sequences for operation in operations}
        if observed & set(RESERVED):
            raise VocabularyError("reserved token inside action operations")
        base = (*RESERVED, *BYTE_TOKENS)
        return cls((*base, *sorted(observed - set(base))))

    def to_dict(self):
        return {"format": VOCAB_FORMAT, "codec": FORMAT, "fit_split": "train",
                "fixed_alphabet": "utf8_bytes_0_255", "tokens": list(self.tokens)}

    @classmethod
    def from_dict(cls, data):
        if data.get("format") != VOCAB_FORMAT or data.get("codec") != FORMAT or data.get("fit_split") != "train":
            raise VocabularyError("unsupported typed-action vocabulary binding")
        return cls(tuple(data["tokens"]))

    def pack(self, operations):
        mapping = {token: index for index, token in enumerate(self.tokens)}
        unknown = sorted({operation["token"] for operation in operations} - set(mapping))
        if unknown:
            raise VocabularyError("validation-only structural tokens: " + ", ".join(unknown[:16]))
        return {"token_ids": [mapping["<BOS>"], *(mapping[operation["token"]] for operation in operations)],
                "pointer_targets": [-1, *(operation["value"] if operation["token"] in POINTER_TOKENS else -1 for operation in operations)],
                "declaration_slots": [-1, *(operation.get("declaration_slot", -1) for operation in operations)],
                "lexical_scopes": [[], *(operation.get("lexical_scope", []) for operation in operations)],
                "input_local_scopes": [[], *(operation.get("input_local_scope", []) for operation in operations)]}
