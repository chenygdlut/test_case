from __future__ import annotations

from dataclasses import dataclass
import random
import unittest
from typing import List, MutableSequence, Sequence, TypeVar


T = TypeVar("T")


def quicksort(values: Sequence[T]) -> List[T]:
    copied = list(values)
    quicksort_inplace(copied)
    return copied


@dataclass(frozen=True)
class _Range:
    left: int
    right: int


def quicksort_inplace(values: MutableSequence[T]) -> None:
    if len(values) < 2:
        return

    stack: List[_Range] = [_Range(0, len(values) - 1)]

    while stack:
        r = stack.pop()
        left, right = r.left, r.right
        while left < right:
            pivot_index = random.randint(left, right)
            values[pivot_index], values[right] = values[right], values[pivot_index]
            pivot = values[right]

            i = left
            for j in range(left, right):
                if values[j] <= pivot:
                    values[i], values[j] = values[j], values[i]
                    i += 1

            values[i], values[right] = values[right], values[i]

            left_size = i - left
            right_size = right - i

            if left_size < right_size:
                if i + 1 < right:
                    stack.append(_Range(i + 1, right))
                right = i - 1
            else:
                if left < i - 1:
                    stack.append(_Range(left, i - 1))
                left = i + 1


def quicksort_recursive(values: MutableSequence[T]) -> None:
    def _quicksort(low: int, high: int) -> None:
        if low < high:
            pivot_index = random.randint(low, high)
            values[pivot_index], values[high] = values[high], values[pivot_index]
            pivot = values[high]

            i = low
            for j in range(low, high):
                if values[j] <= pivot:
                    values[i], values[j] = values[j], values[i]
                    i += 1

            values[i], values[high] = values[high], values[i]

            _quicksort(low, i - 1)
            _quicksort(i + 1, high)

    _quicksort(0, len(values) - 1)


class TestQuickSort(unittest.TestCase):
    def test_empty(self) -> None:
        self.assertEqual(quicksort([]), [])

    def test_single(self) -> None:
        self.assertEqual(quicksort([1]), [1])

    def test_sorted(self) -> None:
        self.assertEqual(quicksort([1, 2, 3, 4]), [1, 2, 3, 4])

    def test_reverse(self) -> None:
        self.assertEqual(quicksort([4, 3, 2, 1]), [1, 2, 3, 4])

    def test_duplicates(self) -> None:
        self.assertEqual(quicksort([3, 1, 2, 3, 3, 0]), [0, 1, 2, 3, 3, 3])

    def test_negative_numbers(self) -> None:
        self.assertEqual(quicksort([0, -1, 5, -10, 3]), [-10, -1, 0, 3, 5])

    def test_inplace(self) -> None:
        values = [3, 2, 1]
        quicksort_inplace(values)
        self.assertEqual(values, [1, 2, 3])

    def test_recursive(self) -> None:
        values = [3, 2, 1]
        quicksort_recursive(values)
        self.assertEqual(values, [1, 2, 3])

    def test_random_against_sorted(self) -> None:
        random.seed(20260309)
        for _ in range(200):
            values = [random.randint(-50, 50) for _ in range(random.randint(0, 200))]
            self.assertEqual(quicksort(values), sorted(values))


if __name__ == "__main__":
    unittest.main()
