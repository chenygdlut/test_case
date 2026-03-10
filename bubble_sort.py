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