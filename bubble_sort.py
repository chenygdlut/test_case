def bubble_sort(arr):
    n = len(arr)
    # 遍历所有数组元素
    for i in range(n):
        # 最后i个元素已经排好序，不需要再比较
        swapped = False
        for j in range(0, n - i - 1):
            # 如果当前元素大于下一个元素，则交换它们
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        # 如果一次遍历中没有发生交换，则数组已经有序，提前退出
        if not swapped:
            break
    return arr

# 示例用法
if __name__ == "__main__":
    test_arr = [64, 34, 25, 12, 22, 11, 90]
    print("原始数组:", test_arr)
    sorted_arr = bubble_sort(test_arr)
    print("排序后数组:", sorted_arr)

import unittest


class TestBubbleSort(unittest.TestCase):
    def test_empty(self) -> None:
        self.assertEqual(bubble_sort([]), [])

    def test_single(self) -> None:
        self.assertEqual(bubble_sort([1]), [1])

    def test_sorted(self) -> None:
        self.assertEqual(bubble_sort([1, 2, 3, 4]), [1, 2, 3, 4])

    def test_reverse(self) -> None:
        self.assertEqual(bubble_sort([4, 3, 2, 1]), [1, 2, 3, 4])

    def test_duplicates(self) -> None:
        self.assertEqual(bubble_sort([3, 1, 2, 3, 3, 0]), [0, 1, 2, 3, 3, 3])

    def test_negative_numbers(self) -> None:
        self.assertEqual(bubble_sort([0, -1, 5, -10, 3]), [-10, -1, 0, 3, 5])

    def test_inplace(self) -> None:
        values = [3, 2, 1]
        result = bubble_sort(values)
        self.assertEqual(result, [1, 2, 3])

    def test_random_against_sorted(self) -> None:
        import random
        random.seed(20260309)
        for _ in range(200):
            values = [random.randint(-50, 50) for _ in range(random.randint(0, 200))]
            self.assertEqual(bubble_sort(values), sorted(values))


if __name__ == "__main__":
    unittest.main()