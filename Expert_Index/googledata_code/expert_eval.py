import os
import math
import pandas as pd

def calculate_h_index(citation ):
    
    """
    h-index 계산 함수
    :param citations: 논문 인용 횟수 리스트
    :return: h-index
    """
    citation.sort(reverse=True)  # 논문을 내림차순으로 정렬

    h_index = 0
    for i, citation  in enumerate(citation):
        if citation  >= i + 1:
            h_index = i + 1
        else:
            break

    return h_index

def calculate_i10_index(citation):  #i10_index
    
    """
    i10-index 계산 함수
    :param citations: 논문 인용 횟수 리스트
    :return: i10-index
    """
    
    i10_index = sum(1 for citation  in citation  if citation  >= 10)
    return i10_index

def calculate_g_index(citation):  #g_index
    
    """
    g-index 계산 함수
    :param citations: 논문 인용 횟수 리스트
    :return: g-index
    """
    citation .sort(reverse=True)  # 논문을 내림차순으로 정렬

    g_index = 0
    citation_sum=0
    for i, citation in enumerate(citation):
        citation_sum+=citation
        if citation_sum >= (i + 1)**2:
            g_index = i + 1

    return g_index

#Object 타입에서 숫자형 체크용
def is_numeric(value):
    try:
        float(value)  # float으로 변환 가능한 경우 숫자
        return True
    except (ValueError, TypeError):
        return False

def calculate_index(dt, pYears, citations):
    # dt = datetime.datetime.now()#현재날짜
    lct = 0
    scoquality = 0
    
    for i in range(len(pYears)):
        lct += max(round((1-(int(dt)-3-pYears[i])*0.1),2), 0.1)
        scoquality += math.log(int(citations[i])*math.exp((int(pYears[i])-dt)/10)+1)
    
    return lct, scoquality

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_rows', None)

#authors.all 파일 위치 주소
authors_file_path = r"./gsc_data/authors.all"

try:
    with open(authors_file_path, 'r', encoding='cp1252') as file:
        lines = file.read().splitlines()
        data = [line.split('|') for line in lines]
        columns = ["Valid", "Name", "Affiliation", "Email Domain", "Citation Count", "ID"]
        df_a = pd.DataFrame(data,columns=columns) # authors 의 a
        # df_a = df_a.drop(columns=["C6", "C7"])
        print("변환 완료")
except FileNotFoundError:
    print("파일을 찾을 수 없습니다.")
except UnicodeDecodeError:
    print("파일을 ANSI(Windows-1252) 인코딩으로 열 수 없습니다.")
    
df_a_copy = df_a.copy()  # df_a의 복사본 생성

for index, row in df_a_copy.iterrows():
    if pd.isna(row[5]):  
        if pd.isna(row[4]):
            if pd.isna(row[3]):  # 1의 경우
                row[5] = row[2] 
            else:  # 2,3,4 의 경우
                row[5] = row[3]
                if is_numeric(row[2]): # 4의 경우
                    row[4] = row[2]
                    row[2] = None
                    row[3] = None
                elif '.' in str(row[2]):  # 3의 경우
                    row[3] = row[2]
                    row[2] = None
                    row[4] = None
                else: # 2의 경우
                    row[3] = None
        else:  # 5,6,7 의 경우
            row[5] = row[4] 
            if is_numeric(row[3]): # 6,7 의 경우
                row[4] = row[3]
                if '.' in str(row[2]): # 7 의 경우
                    row[3] = row[2]
                    row[2] = None
                else: # 6의 경우
                    
                    row[3] = None
            else: #5의 경우
                row[4] = None
                
#gsc_data\DATA 폴더 경로
folder_path = r"./gsc_data/DATA/"

max_scoquality = 15115.985070107283
max_scoquality_list = []

max_recently = 20762.69999999992
max_recently_list = []

max_rct = 14854
max_rct_list = []

alpha = 0
dt = 2013

# df_a_copy의 각 행에서 1번 열의 값을 추출
for index, row in df_a_copy.iterrows():
    name = row[5]  # 1번 열의 값
   
    # 파일 경로 생성
    file_path = os.path.join(folder_path, name + "_.dat")
   
    # 파일이 존재하는지 확인
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            # 파일의 각 줄을 읽어 리스트로 저장
            lines = file.readlines()
 
        # 각 줄을 나누어서 리스트로 저장
        line_lists = [line.strip().split('|') for line in lines]
       
        rct = 0
        #빈 리스트 생성
        citations = [] #인용수
        years = [] #연도
        scoquality_list = []
        for line_list in line_lists:
            if len(line_list) > 0:
                if int(line_list[1]) >= 1973:
                    citations.append(int(line_list[0]))
                    years.append(int(line_list[1]))
                    rct += 1
                    # print(f'ID:{name}, 연도:{line_list[1]}, 인용:{line_list[0]}')
        # if name == 1:
        #     print(years)
        #     print(citations)
        max_rct_list.append(rct)
        sum_citations = sum(citations)
            
        recently, scoquality = calculate_index(dt, years, citations)
        recently = recently * rct/max_rct
        
        max_scoquality_list.append(scoquality)
        max_recently_list.append(recently)
        norm_scoquality = scoquality/max_scoquality
        norm_recently = recently/max_recently

        score = (norm_scoquality * 50) + (norm_recently * 50)
        # print(f'인용수합:{sum_citations}, 품질:{norm_scoquality}, 최신성:{norm_recently}, 점수{score}')
        #각 index 계산 후 df_a_copy에 새로운 열로 추가
        df_a_copy.at[index,"sum_citation"] = sum_citations
        df_a_copy.at[index,"Paper Count"] = len(citations)
        df_a_copy.at[index,"h-index"] = calculate_h_index(citations)
        df_a_copy.at[index,"i10-index"] = calculate_i10_index(citations)
        df_a_copy.at[index,"g-index"] = calculate_g_index(citations)
        df_a_copy.at[index,"scoquality"] = norm_scoquality
        df_a_copy.at[index,"recently"] = norm_recently
        df_a_copy.at[index,"score"] = score

    else:
        print(f"파일을 찾을 수 없습니다: {file_path}")
       
df_a_copy['ID'] = 'id_' + df_a_copy['ID']
 
# print(max_scoquality1)
print(max(max_scoquality_list))
print(max(max_recently_list))
print(max(max_rct_list))

df_a_copy.head(20)
df_a_copy.to_csv('output_sco50_lct50.csv', index=False)
# df_a_copy.to_csv('output_index.csv', index=True)


folder_path = r"./gsc_data/DATA/"
file_path = os.path.join(folder_path, "ScXat-4AAAAJ_.dat")
author_dict = {}
p_dict = {}
line_lists = []

# 파일이 존재하는지 확인
if os.path.exists(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        # 파일의 각 줄을 읽어 리스트로 저장
        lines = file.readlines()

    # 각 줄을 나누어서 리스트로 저장
    line_lists = [line.strip().split('|') for line in lines]
    line_lists.sort(key=lambda x:x[1])

# print(line_lists)

for citations, years in line_lists:
    if years not in author_dict:
        author_dict[years] = 0
        p_dict[years] = 0
    p_dict[years] += 1
    author_dict[years] += int(citations)

print(list(p_dict.keys()))
print(list(p_dict.values()))
print(list(author_dict.values()))

import pandas as pd

# CSV 파일 경로
raw_data = 'output.csv'
df = pd.read_csv(raw_data)

#지수별로 내림차순 정렬
H_sorted=df.sort_values(by='h-index', ascending=False)
I10_sorted=df.sort_values(by='i10-index', ascending=False)
G_sorted=df.sort_values(by='g-index', ascending=False)
Citation_sorted=df.sort_values(by='Citation Count', ascending=False)
PaperC_sorted=df.sort_values(by='Paper Count', ascending=False)
scoquality_sorted=df.sort_values(by='scoquality', ascending=False)
recentness_sorted=df.sort_values(by='rct', ascending=False)
lct_sorted=df.sort_values(by='lct', ascending=False)
score_sorted=df.sort_values(by='score', ascending=False)

#필요없는 컬럼 삭제
H_sorted=H_sorted.reset_index(drop=True, inplace=False)
I10_sorted=I10_sorted.reset_index(drop=True, inplace=False)
G_sorted=G_sorted.reset_index(drop=True, inplace=False)
Citation_sorted=Citation_sorted.reset_index(drop=True, inplace=False)
PaperC_sorted=PaperC_sorted.reset_index(drop=True, inplace=False)
scoquality_sorted=scoquality_sorted.reset_index(drop=True, inplace=False)
recentness_sorted=recentness_sorted.reset_index(drop=True, inplace=False)
lct_sorted=lct_sorted.reset_index(drop=True, inplace=False)
score_sorted=score_sorted.reset_index(drop=True, inplace=False)

#지수 순위 입력
# num=int(input("숫자 입력 : "))
num=89786

#해당 순위이내의 결과만 남김
H_sorted = H_sorted.head(num)
I10_sorted = I10_sorted.head(num)
G_sorted = G_sorted.head(num)
score_sorted = score_sorted.head(num)

Citation_sorted = Citation_sorted.head(num)
PaperC_sorted = PaperC_sorted.head(num)
# scoquality_sorted = scoquality_sorted.head(num)
# recentness_sorted = recentness_sorted.head(num)

score_sorted.drop(columns=['Valid', 'Affiliation', 'sum_citation','h-index', 'i10-index', 'g-index', 'Citation Count','Email Domain', 'Paper Count'], axis=1, inplace=True)
H_sorted.drop(columns=['Valid', 'Affiliation', 'sum_citation','i10-index', 'g-index', 'Citation Count','Email Domain', 'Paper Count', 'scoquality', 'rct', 'lct', 'score'], axis=1, inplace=True)
I10_sorted.drop(columns=['Valid','Affiliation','sum_citation', 'h-index', 'g-index', 'Citation Count','Email Domain', 'Paper Count', 'scoquality', 'rct', 'lct', 'score'], axis=1, inplace=True)
G_sorted.drop(columns=['Valid','Affiliation','sum_citation','i10-index', 'h-index', 'Citation Count','Email Domain',  'Paper Count', 'scoquality', 'rct','lct', 'score'], axis=1, inplace=True)

score_sorted.columns=['name', 'ID', 'scoquality', 'rct', 'lct', 'score']
H_sorted.columns=['name','ID','h_score']
I10_sorted.columns=['name','ID','i10_score']
G_sorted.columns=['name','ID','g_score']

score_sorted['rank'] = score_sorted['score'].rank(method='min', ascending=False).astype(int)
H_sorted['rank'] = H_sorted['h_score'].rank(method='min', ascending=False).astype(int)
I10_sorted['rank'] = I10_sorted['i10_score'].rank(method='min', ascending=False).astype(int)
G_sorted['rank'] = G_sorted['g_score'].rank(method='min', ascending=False).astype(int)

Combine=[H_sorted, I10_sorted, G_sorted]
Combined_df=pd.concat(Combine, axis=1)

print(score_sorted)