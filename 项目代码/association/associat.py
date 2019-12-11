from collections import defaultdict
import jieba
import re
def cut_words(temp:str):
    '''
    分词的函数
    :param temp: 原文本
    :return:分词结果返回到一个列表
    '''
    wenben = re.sub("[1234567890\s+\.\!\/_；：,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "", temp)

    seg_list = jieba.cut(wenben, cut_all=True)  # 全模式分词
    # seg_list = jieba.cut(wenben,cut_all=False)#精确模式分词
    # seg_list = jieba.cut_for_search(wenben) #搜索引擎模式
    lists = []
    for item in seg_list:
        lists.append(item)

    return lists

def get_counts1(lists:list):
    '''
    统计词频的函数get_counts1(),参数
    :param lists: 切好的词的列表
    :return:每个词出现次数的字典
    '''
    counts = {}
    for item in lists:
        if item in counts:
            counts[item] += 1
        else:
            counts[item] = 1
    return counts


def get_top_counts(counts:dict):
    '''
    按照词频排序
    :param counts: 每个词出现次数的字典
    :return:
    '''
    value_keys = sorted([(count, tz) for tz, count in counts.items()], reverse=True)
    result = {}
    for item in value_keys:
        result[item[1]] = item[0]
    return value_keys
    # return result


def get_lianxiang(temp):
    '''
    获取原始文本的所有汉字的联想词
    :param temp: 原始文本
    :return:
    '''
    wenzhang = re.sub("[1234567890\s+\.\!\/_；：,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "", temp)
    a = cut_words(wenzhang)  # 分词
    b = get_counts1(a)  # 词频统计
    data = {}
    for i in range(len(wenzhang)):
        data[wenzhang[i]] = {}
        for key in get_counts1(a):
            try:
                if wenzhang[i] == key[0]:
                    data[wenzhang[i]][key] = b[key]
            except:
                pass
    print(data)

def associat(str):
    '''
    词联想的函数，输入的参数为输入的词，然后根据词库，按照词频高低，
    输出你想输入的词组，即词联想
    '''
    print("请稍等...")
    wenzhang = re.sub("[1234567890\s+\.\!\/_；：,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "", temp)
    a = cut_words(wenzhang)  # 分词
    b = get_counts1(a)  # 词频统计
    data = {}
    for i in range(len(wenzhang)):
        data[wenzhang[i]] = {}
        for key in get_counts1(a):
            try:
                if wenzhang[i] == key[0]:
                    data[wenzhang[i]][key] = b[key]
            except:
                pass
    if str in data:
        dic=data[str]
        a=get_top_counts(dic)
        b=[]
        for item in a:
            b.append(item[1])
        # print(a)
        print("为您联想的几个词为：",b)
        num=input("请选择:")
        num=int(num)
        print(b[num-1])
        # print("联想结果如下:\n",data[str])
    else:
        print("无结果")
temp=""
f = open('E:\\Git\\HCCR\\项目代码\\association\\text.txt')
line = f.readline()
temp=line
# get_lianxiang1("人")  # 调用获取联想词的函数。
