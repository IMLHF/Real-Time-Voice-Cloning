import pypinyin
from pypinyin.style import register
from .textnorm import textnorm
from .nonstd_pinyin import _nonstd_style
import jieba

jieba.initialize()
register('nonstd', _nonstd_style)


def _get_pinyin(text, std=True, pb=False):
    '''
    Params:
        text: string, normalized sentences
        std: boolean, standard pinyin stylc, default: standard pinyin style
        pb: boolen, prosody label, default: False
    
    Returns:
        pinyin: string
    '''
    if std:
        style = pypinyin.Style.TONE3
    else:
        style = 'nonstd'
    
    text = text.strip()

    punctuation = ', '
    if text[-1:] in list(',.?!'):
        punctuation = text[-1:] + ' '
        text = text[:-1]

    pinyin = []
    if pb:
        for word in jieba.cut(text):
            for p in pypinyin.pinyin(word, style):
                if p[0][-1] not in ['1', '2', '3', '4']:
                    pinyin.append(p[0] + '5')
                else:
                    pinyin.append(p[0])
            pinyin.append('/')
    else:
        for p in pypinyin.pinyin(text, style):
            if p[0][-1] not in ['1', '2', '3', '4']:
                pinyin.append(p[0] + '5')
            else:
                pinyin.append(p[0])

    return ' '.join(pinyin).strip('/').strip() + punctuation


def get_pinyin(text, std=True, pb=True):
    sents = textnorm(text)
    pinyins = []
    for s in sents:
        py = _get_pinyin(s, std, pb)
        py = py.replace(",", " ,").replace(".", " .").replace(";", " ;").replace("!", " !").replace("?", " ?")
        py = py.strip()
        pinyins.append(py)
    pinyin = pinyins
    return pinyin
