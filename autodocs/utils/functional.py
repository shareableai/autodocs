from typing import TypeVar, Iterable
T = TypeVar('T')

def skip(item: Iterable[T], n: int) -> Iterable[T]:
    idx = 0
    for element in item:
        if idx < n:
            idx += 1
            continue
        yield element
