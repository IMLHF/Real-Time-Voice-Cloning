from textnorm import get_pinyin


def dummy_test():
    assert get_pinyin('你好', 'ni3 hao3')
