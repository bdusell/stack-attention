from collections.abc import Callable

from .base import Vocabulary, VocabularyBuilder

class ToStringVocabulary(Vocabulary):

    def __init__(self, reserved_names: list[tuple[str, int]]):
        super().__init__()
        self._reserved_names = reserved_names
        for name, index in reserved_names:
            setattr(self, f'{name}_index', index)

    def __add__(self, other: 'ToStringVocabulary') -> 'ConcatenatedToStringVocabulary':
        return ConcatenatedToStringVocabulary(self, other)

    def to_string(self, index: int) -> str:
        raise NotImplementedError

class ToStringVocabularyBuilder(VocabularyBuilder[ToStringVocabulary]):

    def content(self, tokens: list[str]) -> ToStringVocabulary:
        return ContentToStringVocabulary(tokens)

    def catchall(self, token: str) -> ToStringVocabulary:
        return CatchallToStringVocabulary(token)

    def reserved(self, tokens: list[str]) -> ToStringVocabulary:
        return ReservedToStringVocabulary(tokens)

class ContentToStringVocabulary(ToStringVocabulary):

    def __init__(self, tokens: list[str]):
        super().__init__([])
        self._string_list = tokens

    def __len__(self) -> int:
        return len(self._string_list)

    def to_string(self, index: int) -> str:
        return self._string_list[index]

class CatchallToStringVocabulary(ToStringVocabulary):

    def __init__(self, token: str):
        super().__init__([(token, 0)])
        self._token = token

    def __len__(self) -> int:
        return 1

    def to_string(self, index: int) -> str:
        if index == 0:
            return f'<{self._token}>'
        else:
            raise IndexError

class ReservedToStringVocabulary(ToStringVocabulary):

    def __init__(self, tokens: list[str]):
        super().__init__([(name, i) for i, name in enumerate(tokens)])
        self._tokens = tokens

    def __len__(self) -> int:
        return len(self._tokens)

    def to_string(self, index: int) -> str:
        return f'<{self._tokens[index]}>'

class ConcatenatedToStringVocabulary(ToStringVocabulary):

    def __init__(self, first: ToStringVocabulary, second: ToStringVocabulary):
        first_size = len(first)
        super().__init__([
            *first._reserved_names,
            *((name, first_size + index) for name, index in second._reserved_names)
        ])
        self._first = first
        self._second = second

    def __len__(self) -> int:
        return len(self._first) + len(self._second)

    def to_string(self, index: int) -> str:
        first_size = len(self._first)
        if index < 0:
            raise IndexError
        elif index < first_size:
            return self._first.to_string(index)
        else:
            return self._second.to_string(index - first_size)

def build_to_string_vocabulary(
    func: Callable[[ToStringVocabularyBuilder], ToStringVocabulary]
) -> ToStringVocabulary:
    return func(ToStringVocabularyBuilder())
