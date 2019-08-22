import string, sys, os
import gzip, re
from . import hanzi

'''
Symbol definitions
'''
# mapping from 0, 1, ..., 9 to chinese character
digits = {u'0': u'\u96f6', u'1': u'\u4e00', u'2': u'\u4e8c', u'3': u'\u4e09', u'4': u'\u56db', u'5': u'\u4e94', u'6': u'\u516d', u'7': u'\u4e03', u'8': u'\u516b', u'9': u'\u4e5d'}
# mapping from 10, 10^2, 10^3, 10^4, 10^8 to chinese characters
tens = [u'', u'\u5341', u'\u767e', u'\u5343']
thousds = [u'', u'\u4e07', u'\u4ebf', u'\u5146']

# Chinese coded English letters and numbers
chnalpupper = [u'\uff21',u'\uff22', u'\uff23', u'\uff24', u'\uff25', u'\uff26', u'\uff27',u'\uff28', u'\uff29', u'\uff2a', u'\uff2b', u'\uff2c', u'\uff2d',u'\uff2e', u'\uff2f', u'\uff30', u'\uff31', u'\uff32', u'\uff33', u'\uff34', u'\uff35', u'\uff36', u'\uff37', u'\uff38', u'\uff39', u'\uff3a']
chnalplower = [u'\uff41',u'\uff42', u'\uff43', u'\uff44', u'\uff45', u'\uff46', u'\uff47',u'\uff48', u'\uff49', u'\uff4a', u'\uff4b', u'\uff4c', u'\uff4d',u'\uff4e', u'\uff4f', u'\uff50', u'\uff51', u'\uff52', u'\uff53', u'\uff54', u'\uff55', u'\uff56', u'\uff57', u'\uff58', u'\uff59', u'\uff5a']
chnnumber = [u'\uff10', u'\uff11', u'\uff12', u'\uff13', u'\uff14', u'\uff15', u'\uff16', u'\uff17', u'\uff18', u'\uff19']

''' 
Utility Functions
'''

# regrex patterns
pat_comments = re.compile(u'\(.+\)')##（------）
pat_year = re.compile(u'([^%s]{1,4}\u5e74)' % hanzi.punctuation) 
pat_percent = re.compile(u'(([0-9]+)(\.[0-9]+)?[%\u2030])')
pat_decimal = re.compile(u'([0-9]+)(\.[0-9]+)+') 
#pat_decimal = re.compile(u'(([0-9]+)\.([0-9]+)+)')
pat_number = re.compile(u'(([0-9]+)(\.[0-9]+)?)')
pat_zerostr = re.compile(u'([^0-9](0+)[^0-9])')
pat_sentbnd = re.compile(u'([\uff61\u3002\uff01\uff1b\uff1f!、;?,，……《])')
pat_nonchn = re.compile(u'^[0-9a-z\s]*$')
pat_space = re.compile(u'\s+')
# for this specific purpose,i.e., character LM, all the English words are mapped to A, which will never appear in the selected data
pat_english = re.compile(u'[a-zA-Z][a-zA-Z\-\.\s]*')
pat_alphanumer = re.compile(u'\d*[a-zA-Z][a-zA-Z\d\-\.\s]*') 
pat_dot = re.compile(u'\.')
pat_punc = re.compile(u'[%s\u4e28]' % hanzi.punctuation)
pat_char = re.compile(u'[%s]' % hanzi.characters)
pat_nonchar = re.compile(u'[^%sA]' % (hanzi.characters + ',.!?'))

pat_special1 = re.compile(u'\u2501') 
pat_cell = re.compile(u'((\d{3,4})|\d{3,4}-)?\d{7,8}')
pat_date = re.compile(u'\d{4}-\d{1,2}-\d{1,2}')
math_mark = {u'\u003D': u'\u7b49\u4e8e', #“=”
             u'\u2260': u'\u4e0d\u7b49\u4e8e', #'≠'
             u'\u003C': u'\u5c0f\u4e8e',#'<'
             u'\u003E': u'\u5927\u4e8e',#大于
             u'\u2264': u'\u5c0f\u4e8e\u7b49\u4e8e',#小于等于
             u'\u003C\u003D': u'\u5c0f\u4e8e\u7b49\u4e8e',#<=小于等于
             u'\u2265': u'\u5927\u4e8e\u7b49\u4e8e',#大于等于
             u'\u003E\u003D': u'\u5927\u4e8e\u7b49\u4e8e',#>=大于等于
             u'\u002B': u'\u52a0',#加号
             u'\u002D': u'\u51cf',#减号
             u'\u00D7': u'\u4e58',#中文乘号
             u'\u002A': u'\u4e58',#*乘号
             u'\u00F7': u'\u9664\u4ee5',#除号
            }

# Chinese coded English and numbers
pat_chnAlphaUpper = re.compile(u'[\uff21-\uff3a]')
pat_chnAlphaLower = re.compile(u'[\uff41-\uff4a]')
pat_chnNumber = re.compile(u'[\uff10-\uff19]')

# ignore the sentence
pat_sentIgnore1 = re.compile(u'^A\u622a\u7a3f\u5171\u8a08.*\u5247$')


def replaceComments(sr):
    return pat_comments.sub(u'', sr)


def procYear(sr):
    debug = False
    ressr = u''
    resid = 0
    for mt in pat_year.finditer(sr):
        oldsr = sr[mt.start(1):mt.end(1)]
        ressr += sr[resid:mt.start(1)]
        newsr = u''
        if debug:
            print(u'####'+oldsr)
        for ii in range(mt.start(1), mt.end(1)):
            if sr[ii] in digits.keys():
                newsr += digits[sr[ii]]
            else:
                newsr += sr[ii]
        if debug:
            print(u'====' + newsr)
        if newsr != oldsr:
            ressr += newsr
        else:
            ressr += oldsr
        resid = mt.end(1)
    if resid < len(sr):
        ressr += sr[resid:]

    if debug and resid > 0:
        print(u'####'+sr)
        print(u'===='+ressr)

    return ressr


def proc4DigNum(sr):
    ressr = u''
    j = len(sr)-1
    for i in range(0, len(sr)):
        if sr[j] != u'0':
            ressr = digits[sr[j]]+tens[i]+ressr
        else:
            if ressr == u'': 
                ressr = digits[u'0']
            else:
                if ressr[0] != digits[u'0']:
                    ressr = digits[u'0']+ressr
        j=j-1

    # remove extra 0s at the end
    while len(ressr)>1 and ressr[-1]==digits[u'0']:
        ressr=ressr[:-1]

    return ressr

# different from others, the input is the integer string not the whole sentence
def procInteger(sr):
    # too long, convert bit by bit
    if len(sr) > 16:
        return procNumberString(sr)

    oldsr = sr
    ressr = u''
    i = 0
    curlen = len(oldsr)
    while curlen > 0:
        if curlen >= 4:
            cursr = proc4DigNum(oldsr[-4:])
            oldsr = oldsr[:-4]
        else:
            cursr = proc4DigNum(oldsr)
            oldsr = u''
        if cursr != digits[u'0']:
            ressr = cursr + thousds[i] + ressr
        else:
            if ressr == u'':
                ressr = cursr
            else:
                if ressr[0] != digits[u'0']:
                    ressr = digits[u'0']+ressr
        i = i+1
        curlen = len(oldsr)

    # convert begining "one ten" to "ten"
    if len(ressr) >= 2 and ressr[:2] == u'\u4e00\u5341':
        ressr = ressr[1:]

    # remove extra 0s at the end
    while len(ressr) > 1 and ressr[-1] == digits[u'0']:
        ressr = ressr[:-1]

    # remove extra 0s at the beginning
    while len(ressr) > 1 and ressr[0] == digits[u'0']:
        ressr = ressr[1:]

    return ressr


def procNumberString(sr):
    newsr = u''
    for ii in range(0, len(sr)):
        newsr = newsr+digits[sr[ii]]
    return newsr


def procPercent(sr):
    debug = False
    ressr = u''
    resid = 0
    for mt in pat_percent.finditer(sr):
        ressr += sr[resid:mt.start(1)]

        oldsr = mt.group(1)
        intpart = mt.group(2)
        decpart = mt.group(3)
        if debug:
            print(u'####' + oldsr)
        if oldsr[-1] == u'%':
            newsr = u'\u767e\u5206\u4e4b'+procInteger(intpart)
        elif oldsr[-1] == u'\u2030':#'‰'
            newsr = u'\u5343\u5206\u4e4b'+procInteger(intpart)
        else:
            print('Error in converting percentages!')
            sys.exit(1)
        if decpart is not None:
            newsr += u'\u70b9'+procNumberString(decpart[1:])
        if debug:
            print(u'====' + newsr)

        ressr += newsr
        resid = mt.end(1)

    if resid < len(sr):
        ressr += sr[resid:]

    if debug and resid > 0: 
        print(u'####'+sr)
        print(u'===='+ressr)
    return ressr


def procNumber(sr):
    debug=False
    ressr=u''
    resid=0
    for mt in pat_number.finditer(sr):
        ressr+=sr[resid:mt.start(1)]

        oldsr=mt.group(1)
        intpart=mt.group(2)
        decpart=mt.group(3)
        if debug:
            print(u'####'+oldsr)
        newsr=procInteger(intpart)
        if decpart != None:
            newsr+=u'\u70b9'+procNumberString(decpart[1:])
        if debug:
            print(u'===='+newsr)

        ressr+=newsr
        resid=mt.end(1)

    if resid<len(sr):
        ressr+=sr[resid:]

    if debug and resid >0: 
        print(u'####'+sr)
        print(u'===='+ressr)
    return ressr

def procDecimal(sr):
    debug=False
    ressr=u''
    resid=0
    for mt in pat_decimal.finditer(sr):
        ressr+=sr[resid:mt.start(1)]

        oldsr=mt.group(1)
        intpart=mt.group(2)
        decpart=mt.group(3)
        if debug:
            print(oldsr)
        newsr=procInteger(intpart)+u'\u70b9'+procNumberString(decpart)
        if debug:
            print(newsr)

        ressr+=newsr
        resid=mt.end(1)

    if resid<len(sr):
        ressr+=sr[resid:]

    if debug and resid >0: 
        print(sr)
        print(ressr)
    return ressr

# convert single all 0 strings to chinese characters
def procZeroStr(sr):
    debug=False
    newsr=u''
    for mt in pat_zerostr.finditer(sr):
        if debug:
            print(u'####'+sr[mt.start(1):mt.end(1)])
        newsr=u''
        for i in range(mt.start(2),mt.end(2)):
            newsr=newsr+digits[u'0']
        sr=sr[:mt.start(2)]+newsr+sr[mt.end(2):]
        if debug:
            print(sr[mt.start(1):mt.end(1)])
    if debug and newsr != u'':
        print(sr)
    return sr

# map all English words to ''
def procEnglish(sr):
    debug=False
    if pat_english.search(sr):
        if debug:
            print(sr)
        sr=pat_english.sub(u'',sr)
        sr=pat_alphanumer.sub(u'',sr)
        if debug:
            print(sr)
    return sr

def splitPar(sr):
    debug=False
    lst=pat_sentbnd.split(sr)
    sents=[]
    for i, itm in enumerate(lst):
        if itm in list(',，、；;《<'):
            if len(sents)>0:
                sents[-1] += ', ' 
            else:
                sents.append(", ")
        elif itm in list('｡。……'.strip()):
            if len(sents)>0:
                sents[-1] += '. ' 
            else:
                sents.append(". ")
        elif itm in list('!！'.strip()):
            if len(sents)>0:
                sents[-1] += '! ' 
            else:
                sents.append("! ")
        elif itm in list('?？'.strip()):
            if len(sents)>0:
                sents[-1] += '?'
            else:
                sents.append("? ")
        elif itm != u'':
            itm = procMathMark(itm)
            st=pat_punc.sub(u'', itm)
            st=pat_space.sub(u'', st)
            st=pat_special1.sub(u'\u81f3', st)
            if pat_nonchn.match(st) or pat_sentIgnore1.match(st):
                if debug and st != u'':
                    print(st)
            else:
                sents.append(st)
    return sents

# assume english '.' is dot
def procDot(sr):
    sr=pat_dot.sub(u'\u70b9', sr)
    return sr
    
# replace chinese coded alphanumeric
def procChnAlphaNumber(sr):
    debug=False
    flag=False
    orisr=sr
    for ch in chnalpupper:
        if ch in sr:
            sr=re.sub(ch, chr(65+ord(ch)-65313), sr) # 65 - 'A', 65313 - 'A' in Chinese
            flag=True
    for ch in chnalplower:
        if ch in sr:
            sr=re.sub(ch, chr(97+ord(ch)-65345), sr) # a
            flag=True
    for ch in chnnumber:
        if ch in sr:
            sr=re.sub(ch, chr(48+ord(ch)-65296), sr) # 0
            flag=True
    if u'\ufe6a' in sr:
        sr=re.sub(u'\ufe6a', u'%', sr)
    if u'\u2103' in sr:
        sr=re.sub(u'\u2103', u'\u6444\u6c0f\u5ea6', sr) # celus degree
    if u'\u338f' in sr:
        sr=re.sub(u'\u338f', u'\u5343\u514b', sr) # kilo gram
    if u'\u33a1' in sr:
        sr=re.sub(u'\u33a1', u'\u5e73\u65b9\u7c73', sr) # square meters
    if u'\u25cb' in sr:
        sr=re.sub(u'\u25cb', u'\u96f6', sr) # zero

    if debug and flag:
        print(orisr)
        print(sr)
    return sr


# remove all unknown characters
def removeNonChinese(sr):
    debug = False
    if pat_nonchar.search(sr):
        if debug: 
            print(pat_nonchar.findall(sr))
            print(sr)
        sr=pat_nonchar.sub(u'', sr)
        if debug:
            print(sr)
    return sr

### new added part

def procEnd(sr):
    if u'\u0021' in sr:     #translation '！' from english into chinese
        sr=re.sub(u'\u0021', u'\uFF01', sr)
    if u'\u003F' in sr:     #translation '！' from english in chinese
        sr=re.sub(r'\u003F', u'\uFF1F', sr)
    if u'\u002E' in sr:     #translation '.' from english into chinese
        sr=re.sub(u'\u002E', u'\uFF61', sr)
    if u'\u002C' in sr:    # translation '，'
        sr=re.sub(u'\u33a1', u"\uFF0C", sr) # translation '，'
    return sr

def procDecimal_2(sr):
    #sr = '2345 1.123.1.1.1 34579'
    debug=False
    ressr=u''
    resid=0
    for mt in pat_decimal.finditer(sr):
        ressr += sr[resid:mt.start()]
        if debug:
            print(oldsr)
    #     newsr=intpart +u'\u70b9'+ decpart
        str_list = mt.group().split(".")
        newsr = procNumberString(str_list[0])
        for i in range(1,len(str_list)):
            newsr+=u'\u70b9'+ procNumberString(str_list[i])
        if debug:
            print(newsr)
        ressr += newsr
        resid = mt.end()

    if resid<len(sr):
        ressr+=sr[resid:]
    if debug and resid >0:
        print(sr)
        print(ressr)
    return ressr

def proctel(sr):
    debug = False
    ressr = u''
    resid = 0
    for mt in pat_cell.finditer(sr):
        ressr += sr[resid:mt.start()]
        str_list = mt.group().split("-")
        if len(str_list)==2 :
            newsr = str_list[0]+str_list[1]
        else:
            newsr = str_list
        if debug:
            print(newsr)
        ressr += newsr
        resid = mt.end()
    if resid< len(sr):
        ressr += sr[resid:]
    if debug and resid>0:
        print(sr)
        print(ressr)
    return ressr

def procDate(sr):
    debug = False
    ressr = u''
    resid = 0
    for mt in pat_date.finditer(sr):
        ressr += sr[resid:mt.start()]
        str_list = mt.group().split("-")
        newsr = str_list[0]+u'\u5e74'+str_list[1]+u'\u6708'+str_list[2]+u'\u65e5'
        if debug:
            print(newsr)
        ressr += newsr
        resid = mt.end()
    if resid< len(sr):
        ressr += sr[resid:]
    if debug and resid>0:
        print(sr)
        print(ressr)
    return ressr

def procMathMark(sr):
    newsr = u''
    for ii in range(0, len(sr)):
        if sr[ii] in math_mark.keys():
            newsr = newsr+math_mark[sr[ii]]
        else:
            newsr = newsr+sr[ii]
    return newsr

def detectEmpty(sents):
    news_sents = []
    for st in sents:
        if len(st) >= 1:
            news_sents.append(st)
    return news_sents

def textnorm(unproc=''):
    par=unproc
    par=replaceComments(par)
    par=procDate(par)
    par=procChnAlphaNumber(par)
    par=procYear(par)
    par=procPercent(par)
    par=procDecimal_2(par)
    par=procZeroStr(par)
    par=procEnglish(par)
    par=procDot(par)
    #par=procDate(par)
    par=proctel(par)
    sents=splitPar(par)
    sents = [procNumber(st) for st in sents]
    sents = [removeNonChinese(st) for st in sents]
    # sents = [procEnd(st) for st in sents]
    sents = detectEmpty(sents)
    return sents
