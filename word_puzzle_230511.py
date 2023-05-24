# 生成划句谜题
# 将明日方舟技能名(长度2~10)填入m*n网格中(m,n=6~10),每个格子填一个字
# 要求每个技能名中相邻两个字所在的格子也相邻
# 不要求填满网格,未填的格子填入干扰字
# 随机算法,每次生成的谜题不同,同时返回谜底
# 提供难度选择与自定义设置难度
# By 7alent, 2023/05/11, 以后可能还会更新
# 项目网址: https://github.com/7alent/word_puzzle

import copy
import itertools
import json
import random
import numpy as np
import pandas as pd


# 等差数列
def seq(a, b):
    output = []
    term = a
    while term <= b:
        output.append(term)
        term += 1
    return output


# 卷积函数
def convolution(matrix, weight):
    """将矩阵中每个元素的值替换为该元素与其周围元素的加权和"""
    """df为DataFame,元素为数值"""
    """weight为列表,共9个元素，代表权重"""
    output = copy.deepcopy(matrix)
    r_num = len(matrix[:, 0])
    c_num = len(matrix[0,])
    for r_index, c_index in itertools.product(range(r_num), range(c_num)):
        new = 0
        count_num = 0
        for ii in seq(r_index - 1, r_index + 1):
            for jj in seq(c_index - 1, c_index + 1):
                if 0 <= ii <= r_num - 1 and 0 <= jj <= c_num - 1:
                    new += matrix[ii, jj] * weight[count_num]
                count_num += 1
        output[r_index, c_index] = new
    return output


# 由判别矩阵算出得分矩阵的函数
def empty2cv(empty_m):
    """由empty_mat算出cv_mat"""
    """第一次调用convolution是由empty_mat算出气矩阵(air_mat)"""
    """第二次调用convolution是由air_mat算出得分矩阵(cv_mat)"""
    """air_mat第i行第j列元素表示puzzle中第i行第j列未填字的相邻格子数,类比围棋中的气"""
    """cv_mat第i行第j列元素为air_mat第i行第j列元素的相邻元素和,可视为对puzzle第i行第j列格子的评分，越高越好"""
    """empty_mat中为0的位置,cv_mat同样的位置也为0"""
    output = convolution(convolution(empty_m, [0, 1, 0,
                                               1, 0, 1,
                                               0, 1, 0]), [1, 1, 1,
                                                           1, 0, 1,
                                                           1, 1, 1])
    for r_index in range(len(empty_m[:, 0])):
        for c_index in range(len(empty_m[0,])):
            if not empty_m[r_index, c_index]:
                output[r_index, c_index] = 0
    return output


# 检查谜题是否已经完成的函数
def puzzle_finished(empty_m):
    """依据判别矩阵判断谜题完成情况"""
    """若empty_mat元素全为0,即所有格子已填满,则谜题已完成"""
    """若empty_mat的某行/某列有非0元素相邻,即有两个未填字空格相邻,则谜题未完成"""
    """若empty_mat的每行每列都没有相邻非0元素,即未填字空格均不相邻,则谜题已完成"""
    output = float(sum(sum(empty_m))) <= 0
    if not output:
        for r_index in range(len(empty_m[:, 0])):
            for k in range(len(empty_m[r_index,]) - 1):
                if empty_m[r_index, k] > 0 and empty_m[r_index, k + 1] > 0:
                    output = False
                    break
    if not output:
        for c_index in range(len(empty_m[0,])):
            for k in range(len(empty_m[c_index,]) - 1):
                if empty_m[k, c_index] > 0 and empty_m[k + 1, c_index] > 0:
                    output = False
                    break
    return output


# 查找矩阵中指定元素下标的函数
def get_index(matrix, element):
    return [(r_index, c_index) for r_index, c_index in
            itertools.product(range(len(matrix[:, 0])), range(len(matrix[0,]))) if
            matrix[r_index, c_index] == element]


# 返回矩阵中某位置的元素的相邻元素的函数
def get_neighbor(matrix, index, empty_m):
    r_index = index[0]
    c_index = index[1]
    r_num = len(matrix[:, 0])
    c_num = len(matrix[0,])
    index_list = [(r_index - 1, c_index), (r_index, c_index - 1), (r_index, c_index + 1), (r_index + 1, c_index)]
    output = {'Value': [], 'Index': []}
    for k in index_list:
        if (
                0 <= k[0] < r_num
                and 0 <= k[1] < c_num
                and empty_m[k] > 0
        ):
            output['Value'].append(matrix[k])
            output['Index'].append(k)
    return pd.DataFrame(output)


# 依据得分矩阵随机寻找一条路的函数(没有引入路径长度必须大于1的限制)
def find_path(empty_m, cv_m):
    empty_tmp = copy.deepcopy(empty_m)
    cv_tmp = copy.deepcopy(cv_m)
    max_element = np.max(cv_tmp)
    max_element_index = get_index(cv_tmp, max_element)
    start_point = random.choices(max_element_index)[0]
    output = [start_point]
    empty_tmp[start_point] = 0
    cv_tmp = empty2cv(empty_tmp)
    while True:
        next_point_choice = get_neighbor(cv_tmp, output[-1], empty_tmp)
        if len(next_point_choice) <= 0:
            break
        prob = list(next_point_choice['Value'] / sum(next_point_choice['Value']))
        next_point = random.choices(list(next_point_choice['Index']), prob)[0]
        output.append(next_point)
        empty_tmp[next_point] = 0
        cv_tmp = empty2cv(empty_tmp)
    return output


# 将网格中的干扰字'#','&'随机替换为别的字的函数
def replace_noice_char(puzzle_df, noice_char, answer_d):
    output = copy.deepcopy(puzzle_df)
    usable_string_length = list(answer_d.keys())
    while 1 in usable_string_length:
        usable_string_length.remove(1)
    for r_index in range(len(puzzle_df.iloc[:, 0])):
        for c_index in range(len(puzzle_df.iloc[0,])):
            if puzzle_df.iloc[r_index, c_index] == noice_char:
                replace_string = random.choices(answer_d[random.choices(usable_string_length)[0]])[0]
                output.iloc[r_index, c_index] = random.choices(replace_string)[0]
    return output


# 美观地打印谜题的函数
def print_puzzle(puzzle_df):
    for r_index in range(len(puzzle_df.iloc[:, 0])):
        row = ''
        for each_char in puzzle_df.iloc[r_index,]:
            row += each_char
            row += '   '
        print(row + '\n')


# 随机打乱字符串的函数
def reshuffle_str(string, reshuffle_p):
    """按照一定的概率将字符串打乱(不包括全正序和全倒序情况)"""
    """reshuffle_prob为打乱概率"""
    """字符串长度至少为3"""
    if len(string) < 3:
        return False, string
    if random.uniform(0, 1) >= reshuffle_p:
        return False, string
    reshuffle_list = [
        ''.join(reshuffle_choice)
        for reshuffle_choice in itertools.permutations(string)
    ]
    reshuffle_list.remove(string)
    reshuffle_list.remove(string[::-1])
    return True, random.choices(reshuffle_list)[0]


# 生成将语句按路径填入网格的方案的函数(包含干扰字引入,和已选语句的字序打乱)
def fill_puzzle(path, answer_d, max_noice_len, reshuffle_p):
    """从answer_d字典中挑选语句,生成一种填入路径列表path的方案,该方案为语句列表,包含干扰字符(串)"""
    """max_noice_len为非负整数列表,表示每次选好一个可以填入网格的语句后,最多随机引入多少个干扰字"""
    """reshuffle_p为打乱概率,表示新填入的语句有多大概率随机打乱(打乱后的语句便是干扰字)"""
    used_answer = []
    reshuffle_answer = []
    while True:
        while '' in used_answer:
            used_answer.remove('')
        blank_not_used = len(path) - sum(len(_) for _ in used_answer)
        if blank_not_used <= 1:
            used_answer.extend(['&' for _ in range(blank_not_used)])
            break
        else:
            possible_answer_length = [_ for _ in answer_d.keys() if blank_not_used >= _ > 1]
            while True:
                next_answer = random.choices(answer_d[random.choices(possible_answer_length)[0]])[0]
                repeat_num = 0  # 防止死循环的旗帜变量,表示新选的词恰已选过的次数
                if next_answer in used_answer:
                    repeat_num += 1
                elif repeat_num < 10:
                    reshuffle_tuple = reshuffle_str(next_answer, reshuffle_p)
                    if reshuffle_tuple[0]:
                        reshuffle_answer.append(reshuffle_tuple[1])
                        used_answer.append(reshuffle_tuple[1])
                    else:
                        used_answer.append(next_answer)
                    possible_noice_length = blank_not_used - len(next_answer)
                    if possible_noice_length > 0:
                        used_answer.append(
                            '&' * (random.choices(seq(0, min(possible_noice_length, max_noice_len)))[0]))
                    break
                else:
                    used_answer.extend(['&' for _ in range(blank_not_used)])  # 重试次数repeat_num达到10次,退出循环
                    break
    while '' in used_answer:
        used_answer.remove('')
    return [used_answer, reshuffle_answer]


# 生成谜题的函数
def make_puzzle(r_num, c_num, answer_d, max_noice_len, reshuffle_p):
    """依据参数生成谜题并检验谜题是否合理(至少有1个谜底)"""
    """r_num和c_num为谜题的行数和列数"""
    """answer_d为语句库字典,键为语句长度,值为相同长度语句构成的列表"""
    """max_noice_len为最大随机干扰字长度"""
    """reshuffle_p为打乱语句的概率"""

    key_of_puzzle = []

    while not key_of_puzzle:
        # 生成m*n的网格
        puzzle = [['#' for _ in range(c_num)] for _ in range(r_num)]
        puzzle = pd.DataFrame(puzzle)

        # 初始化判别矩阵与得分矩阵
        empty_mat = np.ones(shape=(r_num, c_num))  # 判别矩阵
        cv_mat = empty2cv(empty_mat)  # 得分矩阵

        # 将语句填入网格
        while not puzzle_finished(empty_mat):
            path_found = find_path(empty_mat, cv_mat)
            if len(path_found) < 2:  # 若可行路径只包含一个点,说明谜题已完成
                break
            string_chosen = fill_puzzle(path_found, answer_d, max_noice_len, reshuffle_p)

            flag = 0
            for each in string_chosen[0]:
                for char in each:
                    puzzle.iloc[path_found[flag]] = char
                    empty_mat[path_found[flag]] = 0
                    flag += 1

            cv_mat = empty2cv(empty_mat)

            answer_chosen = [
                each_string
                for each_string in string_chosen[0]
                if '#' not in each_string
                   and '&' not in each_string
                   and each_string not in string_chosen[1]
            ]
            key_of_puzzle.extend(answer_chosen)

        # 将干扰字填入网格
        while '#' in key_of_puzzle:
            key_of_puzzle.remove('#')
        puzzle = replace_noice_char(puzzle, '#', answer_d)
        while '&' in key_of_puzzle:
            key_of_puzzle.remove('&')
        puzzle = replace_noice_char(puzzle, '&', answer_d)

        # 将网格DataFame的行名和列名设定为空字符,输出更美观
        puzzle.index = ['' for _ in range(len(puzzle.iloc[:, 0]))]
        puzzle.columns = ['' for _ in range(len(puzzle.iloc[0,]))]

        # 判断谜题是否合理(有至少1个谜底)
        if key_of_puzzle:
            return puzzle, key_of_puzzle


# 读取输入并判断输入类型是否为数值和数值大小是否在给定区间内的函数
def input_num(description, a, b, num_type):
    """读取input输入的文本并判断其是否为[a,b]区间内的数值"""
    """description为指导input输入的说明文字"""
    """num_type为数据类型"""
    while True:
        string = input(description)
        try:
            num = num_type(string)
        except ValueError:
            print('\n你输入的数值有误!')
        else:
            if a <= num <= b:
                break
            else:
                print('\n你输入的数值有误!')
    return num


# 输出谜题详细规则的函数
def print_guide():
    print('\n\n划字游戏规则:\n')
    print('1.在汉字矩阵中找到尽可能多的技能名称\n')
    print('2.要求:技能名称中相邻的两个字,在汉字矩阵中的位置也相邻\n')
    print('3.简单示例:\n')
    print('下面两个例子中<灯火不灭>都是合理的谜底\n\n')
    print('字   灯   字   字          字   字   字   字\n\n')
    print('字   火   不   字          字   火   灯   字\n\n')
    print('字   字   灭   字          字   不   灭   字\n\n')
    print('字   字   字   字          字   字   字   字\n\n')
    print('下面两个例子中<磁暴锤>都不是合理的谜底\n\n')
    print('字   字   字   字          字   字   字   磁\n\n')
    print('字   字   锤   字          字   字   爆   字\n\n')
    print('磁   爆   字   字          字   锤   字   字\n\n')
    print('字   字   字   字          字   字   字   字\n\n')
    print('简单来说,如果把找到的技能名连成一条线,得到的应该是一条只有直角的折线\n')
    print('或者说,不能斜着找谜底,因为斜对角不算相邻\n')
    print('找到的谜底从左到右或者从右到左读均可\n')
    print('因此下面两个例子中<灯火不灭>都是合理的谜底\n\n')
    print('字   字   字   字          字   字   字   字\n\n')
    print('灯   火   不   字          灭   不   火   灯\n\n')
    print('字   字   灭   字          字   字   字   字\n\n')
    print('字   字   字   字          字   字   字   字')


# 主函数
if __name__ == '__main__':

    # 导入语句库
    with open("./answer.json", "r", encoding="utf-8") as f:
        answer = json.load(f)

    # 处理语句库
    answer_dict = {}
    for i in answer:
        if len(i) in answer_dict:
            answer_dict[len(i)].append(i)
        else:
            answer_dict[len(i)] = [i]

    # 游玩划字游戏
    while True:
        # 输入谜题参数
        print('\n欢迎游玩<明日方舟>划字谜题!')
        while True:
            chosen_option = input_num(
                '\n\n请选择难度或查看规则\n\n简单(1)   中等(2)   困难(3)   自定义难度(4)   查看规则(5)\n\n请输入选项括号内的数字并回车: ',
                1, 5, int)
            if chosen_option == 1:
                m = 6  # 行数
                n = 6  # 列数
                max_noice_length = 2  # 最大随机干扰字长度
                reshuffle_prob = 0.1  # 打乱语句的概率
                print('\n\n已选择[简单]难度!谜题正在生成......')
                break
            elif chosen_option == 2:
                m = 8
                n = 8
                max_noice_length = 3
                reshuffle_prob = 0.3
                print('\n\n已选择[中等]难度!谜题正在生成......')
                break
            elif chosen_option == 3:
                m = 10
                n = 10
                max_noice_length = 4
                reshuffle_prob = 0.5
                print('\n\n已选择[困难]难度!谜题正在生成......')
                break
            elif chosen_option == 4:
                print('\n\n已选择[自定义难度]!请按要求输入各项谜题参数')
                m = input_num('\n\n请输入谜题的行数(6-10): ', 6, 10, int)
                n = input_num('\n\n请输入谜题的列数(6-10): ', 6, 10, int)
                max_noice_length = input_num('\n\n请输入谜题中干扰字的最大长度(0-5): ', 0, 5, int)
                reshuffle_prob = input_num('\n\n请输入谜题中打乱语句的概率(0-1): ', 0, 1, float)
                print('\n\n谜题参数设置成功!谜题正在生成......')
                break
            elif chosen_option == 5:
                print_guide()

        print(
            '\n\n谜题生成成功!\n\n请在下列汉字中尽可能多地找出<干员的技能名称>!\n\n技能名称中相邻的字在下列汉字中的位置也必须相邻哦!')

        # 生成谜题
        final_puzzle, final_key = make_puzzle(m, n, answer_dict, max_noice_length, reshuffle_prob)

        # 输出谜题与谜底
        print('\n\n谜题为:\n')
        print(f'共有{len(final_key)}个谜底\n')
        print_puzzle(final_puzzle)

        # 查看谜底或提示[正在编]
        clue_list = copy.deepcopy(final_key)  # 未查看的提示
        while True:
            chosen_option = input_num(
                '\n\n请选择查看提示或查看谜底\n\n查看提示(1)   查看谜底(2)\n\n请输入选项括号内的数字并回车: ', 1, 2,
                int)
            if chosen_option == 2:
                print('\n\n谜底为:\n')
                for i in final_key:
                    if '#' not in i and '&' not in i:
                        print(i + '\n')
                break
            elif chosen_option == 1 and clue_list:
                clue = random.choices(clue_list)[0]
                clue_list.remove(clue)
                print('\n提示:' + clue + '    剩余' + str(len(clue_list)) + '次提示次数')
            else:
                print('\n\n提示次数已用完,谜底为:\n')
                for i in final_key:
                    if '#' not in i and '&' not in i:
                        print(i + '\n')
                break

        # 是否继续游玩
        chosen_option = input_num('\n\n是否继续游玩?\n\n继续(1)   退出(2)\n\n请输入选项括号内的数字并回车: ', 1, 2, int)
        if chosen_option == 2:
            print('\n\n感谢游玩!')
            break
