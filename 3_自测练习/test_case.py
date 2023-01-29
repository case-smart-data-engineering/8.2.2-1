#!/usr/bin/env python3
import sys
import torch
from my_solution import test

# 测试用例
def test_solution():
    res = test()
    print(res)
    assert res == [[['为什么不让我爱你'], '歌手', ['欣哲']]]  # 判断输出结果

if __name__ == '__main__':
    test_solution()