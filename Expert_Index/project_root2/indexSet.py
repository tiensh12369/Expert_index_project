import os
import math
import pandas as pd
import numpy as np
from itertools import islice
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_rows', None)

def calculate_ar_index(papers):
    """
    AR-index 계산 함수
    :param papers: 논문 인용 횟수와 연도 리스트 [(citations, year), ...]
    :return: AR-index
    """
    current_year = 2013
    age_weighted_citations = sum(
        citations / (current_year - year+1)
        for citations, year in papers if current_year - year > 0
    )
    return math.sqrt(age_weighted_citations)

def calculate_contemporary_h_index(papers):
    """
    Contemporary h-index 계산 함수
    :param papers: 논문 인용 횟수와 연도 리스트 [(citations, year), ...]
    :param gamma: 연도별 가중치 계수 (기본값: 4)
    :param delta: 연도 보정값 (기본값: 1)
    :return: Contemporary h-index
    """
    gamma=4
    delta=1
    current_year = 2013
    weighted_citations = [
        citations * (gamma / (current_year - year + delta))
        for citations, year in papers
        if current_year - year + delta > 0
    ]
    weighted_citations.sort(reverse=True)
    contemporary_h_index = sum(1 for i, wc in enumerate(weighted_citations) if wc >= i + 1)
    return contemporary_h_index

def calculate_Ha_index(papers):
    """
    Ha_index 계산 함수
    :param papers: 논문 인용 횟수와 연도 리스트 [(citations, year), ...]
    :return: Ha_index
    """
    current_year=2013
    average_citations_list=[]
    for paper in papers:
        if paper[1]>2013:
            continue
        average_citations=paper[0]/(current_year-paper[1]+1)
        average_citations_list.append(average_citations)
    average_citations_list.sort(reverse=True)

    Ha_index=0
    for i in range(len(average_citations_list)):
        if average_citations_list[i] >= Ha_index:
            Ha_index = i + 1
        else:
            break
    return Ha_index

def h_pd_index(papers, current_year):
    """
    ℎ_pd-index 계산 (논문의 연간 평균 인용 수를 기반으로 하는 h-index)
    - papers: [(인용 수, 출판 연도)] 형태의 리스트 (각 논문 별 인용 수 및 출판 연도)
    - current_year: 현재 연도
    """
    avg_citations = [(c / (current_year - y + 1), c) for c, y in papers if (current_year - y + 1) > 0]  
    avg_citations.sort(reverse=True, key=lambda x: x[0])  # 연간 평균 인용 수 기준 내림차순 정렬

    h_pd = 0
    for i, (_, c) in enumerate(avg_citations, 1):
        if c >= i:
            h_pd = i
        else:
            break
    return h_pd


def career_year_h_index_by_avg_citations(papers, current_year):
    """
    Career year h-index by average citations 계산
    - papers: [(인용 수, 출판 연도)] 형태의 리스트
    - current_year: 현재 연도 (예: 2025)
    """
    avg_citations = [(c / (current_year - y + 1), c) for c, y in papers if (current_year - y + 1) > 0]  
    avg_citations.sort(reverse=True, key=lambda x: x[0])  # 평균 인용 수 기준 정렬

    h_cy_avg = 0
    for i, (_, c) in enumerate(avg_citations, 1):
        if c >= i:
            h_cy_avg = i
        else:
            break
    return h_cy_avg


def timed_h_index(papers, current_year, t):
    """
    Timed h-index (ℎ_t) 계산
    - papers: [(인용 수, 출판 연도)] 형태의 리스트
    - current_year: 현재 연도
    - t: 최근 t년 동안의 논문만 고려
    """
    filtered_papers = [(c, y) for c, y in papers if (current_year - y) < t]  # 최근 t년 동안의 논문만 선택
    citations = sorted([c for c, _ in filtered_papers], reverse=True)  # 인용 수 내림차순 정렬

    h_t = 0
    for i, c in enumerate(citations, 1):
        if c >= i:
            h_t = i
        else:
            break
    return h_t


def career_year_h_index_by_publications(papers):
    count_dict = {}
    for line in papers:
        year = line[1]
        if year in count_dict:
            count_dict[year] += 1  
        else:
            count_dict[year] = 1
            
    sorted_list = list(sorted(count_dict.values(), reverse=True))
    # print(sorted_list)
    career_year_h_index_by_p = 0
    for i in range(len(sorted_list)):
        h = i + 1
        if sorted_list[i] >= h:
            career_year_h_index_by_p = h
        else:
            h -= 1
            break
    return career_year_h_index_by_p

def career_year_h_index_by_publications_year_citations(papers):
    career_year_h_index_by_publications_year_c = 0
    for i in range(len(papers)):
        h = i + 1
        if int(papers[i][0]) >= h:
            career_year_h_index_by_publications_year_c = h
        else:
            break
    return career_year_h_index_by_publications_year_c

def calculate_H_index(papers):
    """
    H_index 계산 함수
    :param papers: 논문 인용 횟수와 연도 리스트 [(citations, year), ...]
    :return: H_index
    """
    cit_list=[]
    for paper in papers:
        cit_list.append(paper[0])

    cit_list.sort(reverse=True)

    H_index=0
    for i in range(len(cit_list)):
        if cit_list[i] >= H_index:
            H_index = i + 1
        else:
            break
    return H_index

def calculate_G_index(papers):
    """
    H_index 계산 함수
    :param papers: 논문 인용 횟수와 연도 리스트 [(citations, year), ...]
    :return: H_index
    """
    cit_list=[]
    for paper in papers:
        cit_list.append(paper[0])

    cit_list.sort(reverse=True)

    G_index=0
    for i in range(len(cit_list)):
        if cit_list[i] >= G_index*G_index:
            G_index = i + 1
        else:
            break
    return G_index

def calculate_I10_index(papers):
    """
    H_index 계산 함수
    :param papers: 논문 인용 횟수와 연도 리스트 [(citations, year), ...]
    :return: H_index
    """
    cit_list=[]
    for paper in papers:
        cit_list.append(paper[0])

    cit_list.sort(reverse=True)

    I10_index=0
    for i in range(len(cit_list)):
        if cit_list[i] >= 10:
            I10_index = i + 1
        else:
            break
    return I10_index

def calculate_rec_qual(papers):
    recency_score = 0
    quality_score = 0

    for paper in papers:
        pYears=paper[1]
        citations=paper[0]
        
        current_year=2013
            
        recency = max(round((1 - (int(current_year) - 3 - pYears) * 0.1), 2), 0.1)
        
        if recency > 1:
            recency_score += 1
        else:
            recency_score += recency
        
        quality_score += math.log(int(citations) * math.exp((int(pYears) - current_year) / 10) + 1)
    
    return recency_score, quality_score

def calculate_expert_index(papers, a, b):
    recency_score, quality_score=calculate_rec_qual(papers)
    paper_count=len(papers)
    combined_recency=paper_count*recency_score
    norm_quality_score = quality_score / max_quality if max_quality > 0 else 0
    norm_combined_recency = combined_recency / max_combined_recency if max_combined_recency > 0 else 0
    expert_index = (norm_quality_score * a) + (norm_combined_recency * b)

    return expert_index

author_id_dict = {"Name":[], "ID":[]}
author_file_path = r"./gsc_data/authors.all"
with open(author_file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()
    # print(lines[:10])
    line_list = [line.strip().split('|') for line in lines]
    # print(line_list[:10])

    for line in line_list:
        name = line[1]
        id = line[-1]
        author_id_dict["Name"].append(name)
        author_id_dict["ID"].append(id)

author_id_dict_df=pd.DataFrame(author_id_dict)

print(author_id_dict_df)

folder_path = r"./gsc_data/DATA/"

max_combined_recency=0
max_quality=0
max_paper_count=0
# 데이터 처리 부분
for index, row in author_id_dict_df.iterrows():
    file_path = os.path.join(folder_path, f"{row['ID']}_.dat")
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            line_lists = [line.strip().split('|') for line in lines]
            # 논문 데이터와 저자 수 추출
            papers = []

            paper_count=len(line_lists)

            for line_list in line_lists:
                if len(line_list) >= 2:  # 인용 수, 연도, 저자 수가 있는 경우
                    citations = int(line_list[0])
                    year = int(line_list[1])
                    papers.append((citations, year))

            recency_score, quality_score=calculate_rec_qual(papers)

            max_quality=max(max_quality, quality_score)
            max_paper_count=max(max_paper_count, paper_count)

            combined_recency=paper_count*recency_score
            max_combined_recency=max(max_combined_recency, combined_recency)


folder_path = r"./gsc_data/DATA/"

# 데이터 처리 부분
for index, row in author_id_dict_df.iterrows():
    file_path = os.path.join(folder_path, f"{row['ID']}_.dat")
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            line_lists = [line.strip().split('|') for line in lines]
            # 논문 데이터와 저자 수 추출
            papers = []

            for line_list in line_lists:
                if len(line_list) >= 2:  # 인용 수, 연도, 저자 수가 있는 경우
                    citations = int(line_list[0])
                    year = int(line_list[1])
                    papers.append((citations, year))
            a=50
            b=50
            # 각 지수 계산
            author_id_dict_df.at[index, "h_index"] = calculate_H_index(papers)
            author_id_dict_df.at[index, "g_index"] = calculate_G_index(papers)
            author_id_dict_df.at[index, "i10_index"] = calculate_I10_index(papers)
            author_id_dict_df.at[index, "ar_index"] = calculate_ar_index(papers)
            author_id_dict_df.at[index, "contemporary_h_index"] = calculate_contemporary_h_index(papers)
            author_id_dict_df.at[index, "Ha_index"] = calculate_Ha_index(papers)
            author_id_dict_df.at[index, "career_year_h_index_by_publications"] = career_year_h_index_by_publications(papers)
            author_id_dict_df.at[index, "career_year_h_index_by_publications_year_citations"] = career_year_h_index_by_publications_year_citations(papers)
            author_id_dict_df.at[index, "expert_index"] = calculate_expert_index(papers,a ,b)


# 정렬 및 순위 계산
df_sorted = author_id_dict_df.copy()
 
# 각 지수별 정렬
H_sorted = df_sorted.sort_values(by='h_index', ascending=False)
G_sorted = df_sorted.sort_values(by='g_index', ascending=False)
I10_sorted = df_sorted.sort_values(by='i10_index', ascending=False)
AR_sorted = df_sorted.sort_values(by='ar_index', ascending=False)
CH_sorted = df_sorted.sort_values(by='contemporary_h_index', ascending=False)
Ha_sorted = df_sorted.sort_values(by='Ha_index', ascending=False)
HP_sorted = df_sorted.sort_values(by='career_year_h_index_by_publications', ascending=False)
HPYC_sorted = df_sorted.sort_values(by='career_year_h_index_by_publications_year_citations', ascending=False)
expert_sorted=df_sorted.sort_values(by='expert_index', ascending=False)

# 순위 계산
H_sorted['rank'] = H_sorted['h_index'].rank(method='min', ascending=False).astype(int)
G_sorted['rank'] = G_sorted['g_index'].rank(method='min', ascending=False).astype(int)
I10_sorted['rank'] = I10_sorted['i10_index'].rank(method='min', ascending=False).astype(int)
AR_sorted['rank'] = AR_sorted['ar_index'].rank(method='min', ascending=False).astype(int)
CH_sorted['rank'] = CH_sorted['contemporary_h_index'].rank(method='min', ascending=False).astype(int)
Ha_sorted['rank'] = Ha_sorted['Ha_index'].rank(method='min', ascending=False).astype(int)
HP_sorted['rank'] = HP_sorted['career_year_h_index_by_publications'].rank(method='min', ascending=False).astype(int)
HPYC_sorted['rank'] = HPYC_sorted['career_year_h_index_by_publications_year_citations'].rank(method='min', ascending=False).astype(int)
expert_sorted['rank'] = expert_sorted['expert_index'].rank(method='min', ascending=False).astype(int)


# 결과 저장
# df_sorted.to_csv('output_with_new_indices.csv', index=False)
print(expert_sorted[['Name','ID', 'rank']])

h_lists = H_sorted['ID'].values.tolist()
g_lists = G_sorted['ID'].values.tolist()
i10_lists = I10_sorted['ID'].values.tolist()
ar_lists = AR_sorted['ID'].values.tolist()
ch_lists = CH_sorted['ID'].values.tolist()
ha_lists = Ha_sorted['ID'].values.tolist()
hp_lists = HP_sorted['ID'].values.tolist()
hpyc_lists = HPYC_sorted['ID'].values.tolist()
expert_lists=expert_sorted['ID'].values.tolist()

print(h_lists[:10])

def count_same_index_values(list1, list2):
    # 두 리스트의 길이가 다를 경우, 짧은 리스트의 길이만큼만 비교
    min_length = min(len(list1), len(list2))
    count = 0

    for i in range(min_length):
        if list1[i][0] == list2[i][0] and list1[i][2]==list2[i][2]:
            count += 1

    return count

def count_duplicates(list1, list2):
    # 두 리스트의 집합을 생성하여 중복된 요소를 찾음
    set1 = set(list1)
    set2 = set(list2)
    
    # 교집합을 통해 중복된 요소를 찾고 그 개수를 반환
    duplicates = set1.intersection(set2)
    return len(duplicates)

# 상위 n% 저자간의 중복성 
# 1% 898명, 2% 1796명, 3% 2684명, 4% 3590명, 5% 4490명
author_num=3590

index_list=[]
index_list.append(h_lists[:author_num])
index_list.append(g_lists[:author_num])
index_list.append(i10_lists[:author_num])
index_list.append(ar_lists[:author_num])
index_list.append(ch_lists[:author_num])
index_list.append(ha_lists[:author_num])
index_list.append(hp_lists[:author_num])
index_list.append(hpyc_lists[:author_num])
index_list.append(expert_lists[:author_num])

rbo_matrix={'index':['h_index', 'g_index', 'i10_index', 'ar_index', 'ch_index', 'ha_index', 'hp_index', 'hpyc_index','expert_index'],
            'h_index':[0,0,0,0,0,0,0,0],
            'g_index':[0,0,0,0,0,0,0,0],
            'i10_index':[0,0,0,0,0,0,0,0],
            'ar_index':[0,0,0,0,0,0,0,0],
            'ch_index':[0,0,0,0,0,0,0,0],
            'ha_index':[0,0,0,0,0,0,0,0],
            'hp_index':[0,0,0,0,0,0,0,0],
            'hpyc_index':[0,0,0,0,0,0,0,0],
            'expert_index':[0,0,0,0,0,0,0,0]
            }

# matrix=[[0,0,0,0,0,0,0,0],
#         [0,0,0,0,0,0,0,0],
#         [0,0,0,0,0,0,0,0],
#         [0,0,0,0,0,0,0,0],
#         [0,0,0,0,0,0,0,0],
#         [0,0,0,0,0,0,0,0],
#         [0,0,0,0,0,0,0,0],
#         [0,0,0,0,0,0,0,0],
#         [0,0,0,0,0,0,0,0]
#         ]

# for i in range(len(index_list)):
#     for k in range(len(index_list)):
#         matrix[i][k]=count_duplicates(index_list[i], index_list[k])/4490

# matrix를 올바른 차원으로 초기화
matrix = [[0 for _ in range(len(index_list))] for _ in range(len(index_list))]  # matrix 크기 설정

for i in range(len(index_list)):
    for k in range(len(index_list)):
        matrix[i][k] = count_duplicates(index_list[i], index_list[k]) / author_num


print(matrix[0])

rbo_matrix['h_index']=matrix[0]
rbo_matrix['g_index']=matrix[1]
rbo_matrix['i10_index']=matrix[2]
rbo_matrix['ar_index']=matrix[3]
rbo_matrix['ch_index']=matrix[4]
rbo_matrix['ha_index']=matrix[5]
rbo_matrix['hp_index']=matrix[6]
rbo_matrix['hpyc_index']=matrix[7]
rbo_matrix['expert_index']=matrix[8]


print(rbo_matrix)