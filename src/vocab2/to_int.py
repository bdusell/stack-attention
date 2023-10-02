from collections.abc import Callable

from .base import Vocabulary, VocabularyBuilder

class ToIntVocabulary(Vocabulary):

    def __add__(self, other: 'ToIntVocabulary') -> 'ConcatenatedToIntVocabulary':
        return ConcatenatedToIntVocabulary(self, other)

    def to_int(self, token: str) -> int:
        raise NotImplementedError

    def has_catchall(self) -> bool:
        raise NotImplementedError

class ToIntVocabularyBuilder(VocabularyBuilder[ToIntVocabulary]):

    def content(self, tokens: list[str]) -> ToIntVocabulary:
        return ContentToIntVocabulary(tokens)

    def catchall(self, token: str) -> ToIntVocabulary:
        return CatchallToIntVocabulary()

    def reserved(self, tokens: list[str]) -> ToIntVocabulary:
        return ReservedToIntVocabulary(len(tokens))

class ContentToIntVocabulary(ToIntVocabulary):

    def __init__(self, tokens: list[str]):
        super().__init__()
        self._string_to_int = { s : i for i, s in enumerate(tokens) }

    def __len__(self) -> int:
        return len(self._string_to_int)

    def to_int(self, token: str) -> int:
        return self._string_to_int[token]

    def has_catchall(self) -> bool:
        return False

class CatchallToIntVocabulary(ToIntVocabulary):

    def __len__(self) -> int:
        return 1

    def to_int(self, token: str) -> int:
        return 0

    def has_catchall(self) -> bool:
        return True

class ReservedToIntVocabulary(ToIntVocabulary):

    def __init__(self, size: int):
        super().__init__()
        self._size = size

    def __len__(self) -> int:
        return self._size

    def to_int(self, token: str) -> int:
        raise KeyError

    def has_catchall(self) -> bool:
        return False

class ConcatenatedToIntVocabulary(ToIntVocabulary):

    def __init__(self, first: ToIntVocabulary, second: ToIntVocabulary):
        super().__init__()
        self._first = first
        self._second = second
        if self._first.has_catchall():
            if self._second.has_catchall():
                raise ValueError('vocabulary has multiple catchalls')
            else:
                prioritize_first = False
        else:
            prioritize_first = self._second.has_catchall()
        self._check_first = self._first_to_int if prioritize_first else self._second_to_int
        self._check_second = self._second_to_int if prioritize_first else self._first_to_int

    def __len__(self) -> int:
        return len(self._first) + len(self._second)

    def to_int(self, token: str) -> int:
        try:
            return self._check_first(token)
        except KeyError:
            return self._check_second(token)

    def has_catchall(self) -> bool:
        return self._first.has_catchall() or self._second.has_catchall()

    def _first_to_int(self, token):
        return self._first.to_int(token)

    def _second_to_int(self, token):
        return len(self._first) + self._second.to_int(token)

def build_to_int_vocabulary(
    func: Callable[[ToIntVocabularyBuilder], ToIntVocabulary]
) -> ToIntVocabulary:
    return func(ToIntVocabularyBuilder())
