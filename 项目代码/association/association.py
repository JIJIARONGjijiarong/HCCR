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


def find(str):
    f = open('E:\Git\HCCR\项目代码\\association\\temp.txt','r')
    a = f.read()
    data = eval(a)
    if str in data:
        dic=data[str]
        a=get_top_counts(dic)
        b=[]
        for item in a:
            b.append(item[1])
        print("为您联想的几个词为：",b)
        num=input("请选择:")
        num=int(num)
        print(b[num-1])
    else:
        print("无结果")