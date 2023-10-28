import matplotlib.pyplot as plt
import os
def shapley_class(args,part,part_count, task, class_name, num_correct, nb_data, save_path):
    plt.clf()
    data2={}
    for i in range(7):
        data2[part[i]]=(list(part_count.values())[i])/nb_data
    # {0: 1139, 1: 5, 2: 3, 3: 47, 4: 5, 5: 5, 6: 2}
    # 딕셔너리의 key와 value를 각각 리스트로 추출
    x = list(data2.keys())
    y = list(data2.values())

    # 그래프 생성
    plt.bar(x, y)

    # x축과 y축에 라벨 추가
    plt.xlabel('x')
    plt.ylabel('y')

    # 그래프 제목 추가
    # plt.title(f'{num_correct}/{len(dataset)}={num_correct/len(dataset)*100}%')
    plt.title(f'{task} task  : {class_name} samples\n{num_correct}/{nb_data}={num_correct/nb_data*100:.2f}%')
    # 그래프 표시
    plt.savefig(os.path.join(save_path,f'{task}_{class_name}_{args.size}_{args.norm_type}.png'))
    
    
def shapley_task(args,part,part_count_list, task, class_name, num_correct, nb_data, save_path):
    plt.clf()
    data2={}
    for part_name in part:
        data2[part_name]=0
    for i in range(7):
        value=0
        for part_count in part_count_list:
            value+=(list(part_count.values())[i])
        data2[part[i]]=value / nb_data
    # {0: 1139, 1: 5, 2: 3, 3: 47, 4: 5, 5: 5, 6: 2}
    # 딕셔너리의 key와 value를 각각 리스트로 추출
    x = list(data2.keys())
    y = list(data2.values())

    # 그래프 생성
    plt.bar(x, y)

    # x축과 y축에 라벨 추가
    plt.xlabel('x')
    plt.ylabel('y')

    # 그래프 제목 추가
    # plt.title(f'{num_correct}/{len(dataset)}={num_correct/len(dataset)*100}%')
    plt.title(f'{task} task total  : \n{num_correct}/{nb_data}={num_correct/nb_data*100:.2f}%')
    # 그래프 표시
    plt.savefig(os.path.join(save_path,f'{task}_total_{args.size}_{args.norm_type}.png'))