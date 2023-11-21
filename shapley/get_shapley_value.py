
import torch
from itertools import product
import math




def binary_to_decimal(binary_tuple):
    decimal_value = 0
    binary_length = len(binary_tuple)

    for i, bit in enumerate(binary_tuple):
        decimal_value += bit * (2 ** (binary_length - i - 1))

    return decimal_value
def decimal_to_binary(decimal_value, num_bits):
    binary_tuple = []
    
    for i in range(num_bits):
        bit = (decimal_value >> (num_bits - i - 1)) & 1
        binary_tuple.append(bit)
    
    return tuple(binary_tuple)
def count_ones(binary_tuple):
    count = 0
    for bit in binary_tuple:
        if bit == 1:
            count += 1
    return count
def get_ordered_pair():

    n = 6  # digit의 개수
    digits = [0, 1]  # 각 digit의 가능한 값

    # 경우의 수 생성
    part_combinations = list(product(digits, repeat=n))


    index_to_insert = 1  # 두 번째 위치에 추가하려면 인덱스 1을 사용합니다.
    all_ordered_pair=[]
    for index in range(7):
        ordered_pair=[] 
        index_to_insert = index
        for combi in part_combinations:
            insert_value = [0,1]
            new_combi_0= combi[:index_to_insert] + (insert_value[0],) + combi[index_to_insert:]
            new_combi_1= combi[:index_to_insert] + (insert_value[1],) + combi[index_to_insert:]
            ordered_pair.append([binary_to_decimal(new_combi_0),binary_to_decimal(new_combi_1)])
        all_ordered_pair.append(ordered_pair)
    all_ordered_pair=torch.Tensor(all_ordered_pair)
    num_part = (all_ordered_pair.shape[0])
    num_case = (all_ordered_pair.shape[1])
    weights = torch.zeros((num_part,num_case))
    for i in range(num_part):
        for j in range(num_case):
            # all_ordered_pair의 값 가져오기
            value = int(all_ordered_pair[i, j, 1])
            
            # 이진수로 변환
            binary_value = decimal_to_binary(value, 7)
            
            # 1의 개수 세기
            num_ones = binary_value.count(1)
            
            # num * (7 combination num) 계산
            combination = math.comb(num_part,num_ones)
            weight = num_ones * combination
            
            # 결과를 weights에 저장
            weights[i, j] = weight
    return all_ordered_pair, weights

    
    
def get_shapley_matrix(all_ordered_pair, correct_output):
    shapley_values = torch.zeros_like(all_ordered_pair, dtype=torch.float32)

    # 각 ordered pair에 대한 값을 가져와 shapley_values에 저장
    for a,ordered_pairs in enumerate(all_ordered_pair):
        for i, ordered_pair in enumerate(ordered_pairs):
            # ordered_pair를 인덱스로 사용하여 correct_output에서 값을 가져옴
            indices = ordered_pair  # ordered_pair를 텐서로 변환
            # print(indices)
            values1 = correct_output[int(indices[0])]
            values2 = correct_output[int(indices[1])]  # correct_output에서 해당 위치의 값 가져오기
            # print(values1,values2)
            shapley_values[a,i] = torch.cat([values1.unsqueeze(0),values2.unsqueeze(0)],dim=0)
    return shapley_values
def binary_to_decimal(binary_tuple):
    decimal_value = 0
    binary_length = len(binary_tuple)

    for i, bit in enumerate(binary_tuple):
        decimal_value += bit * (2 ** (binary_length - i - 1))

    return decimal_value
def decimal_to_binary(decimal_value, num_bits):
    binary_tuple = []
    
    for i in range(num_bits):
        bit = (decimal_value >> (num_bits - i - 1)) & 1
        binary_tuple.append(bit)
    
    return tuple(binary_tuple)
def count_ones(binary_tuple):
    count = 0
    for bit in binary_tuple:
        if bit == 1:
            count += 1
    return count
def get_ordered_pair(i=7):

    n = i-1  # digit의 개수
    digits = [0, 1]  # 각 digit의 가능한 값

    # 경우의 수 생성
    part_combinations = list(product(digits, repeat=n))


    index_to_insert = 1  # 두 번째 위치에 추가하려면 인덱스 1을 사용합니다.
    all_ordered_pair=[]
    for index in range(i):
        ordered_pair=[] 
        index_to_insert = index
        for combi in part_combinations:
            insert_value = [0,1]
            new_combi_0= combi[:index_to_insert] + (insert_value[0],) + combi[index_to_insert:]
            new_combi_1= combi[:index_to_insert] + (insert_value[1],) + combi[index_to_insert:]
            ordered_pair.append([binary_to_decimal(new_combi_0),binary_to_decimal(new_combi_1)])
        all_ordered_pair.append(ordered_pair)
    all_ordered_pair=torch.Tensor(all_ordered_pair)
    num_part = (all_ordered_pair.shape[0])
    num_case = (all_ordered_pair.shape[1])
    weights = torch.zeros((num_part,num_case))
    for i in range(num_part):
        for j in range(num_case):
            # all_ordered_pair의 값 가져오기
            value = int(all_ordered_pair[i, j, 1])
            
            # 이진수로 변환
            binary_value = decimal_to_binary(value, 7)
            
            # 1의 개수 세기
            num_ones = binary_value.count(1)
            
            # num * (7 combination num) 계산
            combination = math.comb(num_part,num_ones)
            weight = num_ones * combination
            
            # 결과를 weights에 저장
            weights[i, j] = weight
    return all_ordered_pair, weights


def get_shapley_matrix(all_ordered_pair, correct_output):
    shapley_values = torch.zeros_like(all_ordered_pair, dtype=torch.float32)

    # 각 ordered pair에 대한 값을 가져와 shapley_values에 저장
    for a,ordered_pairs in enumerate(all_ordered_pair):
        for i, ordered_pair in enumerate(ordered_pairs):
            # ordered_pair를 인덱스로 사용하여 correct_output에서 값을 가져옴
            indices = ordered_pair  # ordered_pair를 텐서로 변환
            # print(indices)
            values1 = correct_output[int(indices[0])]
            values2 = correct_output[int(indices[1])]  # correct_output에서 해당 위치의 값 가져오기
            # print(values1,values2)
            shapley_values[a,i] = torch.cat([values1.unsqueeze(0),values2.unsqueeze(0)],dim=0)
    return shapley_values