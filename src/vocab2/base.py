from typing import Generic, TypeVar

class Vocabulary:

    def __len__(self) -> int:
        raise NotImplementedError

V = TypeVar('V')

class VocabularyBuilder(Generic[V]):

    def content(self, tokens: list[str]) -> V:
        raise NotImplementedError

    def catchall(self, token: str) -> V:
        raise NotImplementedError

    def reserved(self, tokens: list[str]) -> V:
        raise NotImplementedError
