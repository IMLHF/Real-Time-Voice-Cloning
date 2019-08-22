from pypinyin.style._utils import get_initials
from pypinyin.style.finals import converter as finals_converter


sil_finals_initials = ['zh', 'ch', 'sh', 'r', 'z', 'c', 's']
sil_finals = ['i{}'.format(t) for t in range(1, 6)]
sil_finals.append('i')


def _nonstd_style(pinyin, **kwargs):
    initials = get_initials(pinyin, strict=True)
    finals = finals_converter.to_finals_tone3(pinyin, strict=True)

    # process silent finals
    if finals in sil_finals and initials in sil_finals_initials:
        finals = finals.replace('i', '')

    pinyin = '' + initials + finals

    return pinyin