from pinyin import get_pinyin


def dummy_test():
    assert get_pinyin('你好', 'ni3 hao3')

if __name__ == "__main__":
    print(get_pinyin("你好？中文！中文的，符号"))
