import numpy as np
import pandas as pd
import jieba.analyse
import os


def get_file_keywords(dir):
    data_array = []
    set_word = []

    try:
        fo = open('dic_test.txt', 'w+', encoding='UTF-8')
        for home, dirs, files in os.walk(dir):
            for filename in files:
                fullname = os.path.join(home, filename)
                f = open(fullname, 'r', encoding='UTF-8')
                sentence = f.read()
                word = " ".join(jieba.analyse.extract_tags(sentence=sentence,
                                                           topK=30,
                                                           withWeight=False,
                                                           allowPOS=('n')))
                words = words.split(' ')
                data_array.append(words)
                for word in words:
                    if word not in set_word:
                        set_word.append(word)
        set_word = list(set(set_word))
    except Exception as reason:
        print('Wrong:', reason)
        return data_array, set_word


def build_matrix(set_word):
    edge = len(set_word) + 1  # 建立矩阵，矩阵的高度和宽度为关键词集合的长度+1
    # '''matrix = np.zeros((edge, edge), dtype=str)''' # 另一种初始化方法
    matrix = [['' for j in range(edge)] for i in range(edge)]  # 初始化矩阵
    matrix[0][1:] = np.array(set_word)
    matrix = list(map(list, zip(*matrix)))
    matrix[0][1:] = np.array(set_word)  # 赋值矩阵的第一行与第一列
    return matrix


def count_matrix(matrix, formated_data):
    for row in range(1, len(matrix)):
    # 遍历矩阵第一行，跳过下标为0的元素
        for col in range(1, len(matrix)):
        # 遍历矩阵第一列，跳过下标为0的元素
        # 实际上就是为了跳过matrix中下标为[0][0]的元素，因为[0][0]为空，不为关键词
        if matrix[0][row] == matrix[col][0]:
            # 如果取出的行关键词和取出的列关键词相同，则其对应的共现次数为0，即矩阵对角线为0
            matrix[col][row] = str(0)
        else:
            counter = 0 # 初始化计数器
            for ech in formated_data:
                # 遍历格式化后的原始数据，让取出的行关键词和取出的列关键词进行组合，
                # 再放到每条原始数据中查询
                if matrix[0][row] in ech and matrix[col][0] in ech:
                    counter += 1
                else:
                    continue
            matrix[col][row] = str(counter)
    return matrix


def main():
    format_data, set_word = get_file_keywords(r'D:\untitled\test')
    print(set_word)
    print(format_data)
    matrix = build_matrix(set_word)
    matrix = count_matrix(matrix, format_data)
    data1 = pd.DataFrame(matrix)
    data1.to_csv('data.csv', index=0, columns=None, encoding='utf_8_sig')


main()

