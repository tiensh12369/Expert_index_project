import math

def h_index(papers):
    """
    H-index 계산 함수
    """
    citations = sorted([c for c, _ in papers], reverse=True)
    return sum(1 for i, c in enumerate(citations, 1) if c >= i)

def g_index(papers):
    """
    G-index 계산 함수
    """
    citations = sorted([c for c, _ in papers], reverse=True)
    total_citations = 0
    g_index = 0
    
    for i, c in enumerate(citations, 1):
        total_citations += c
        if total_citations >= i * i:
            g_index = i
        else:
            break
    return g_index

def i10_index(papers):
    """
    I10-index 계산 함수
    """
    return sum(1 for c, _ in papers if c >= 10)

def ar_index(papers, current_year):
    """
    AR-index 계산 함수
    """
    age_weighted_citations = sum(
        citations / (current_year - year + 1)
        for citations, year in papers if current_year - year > 0
    )
    return math.sqrt(age_weighted_citations)

def contemporary_h_index(papers, current_year, gamma=4, delta=1):
    """
    Contemporary h-index 계산 함수
    """
    weighted_citations = [
        citations * (gamma / (current_year - year + delta))
        for citations, year in papers
        if current_year - year + delta > 0
    ]
    weighted_citations.sort(reverse=True)
    contemporary_h_index = sum(1 for i, wc in enumerate(weighted_citations) if wc >= i + 1)
    return contemporary_h_index

def ha_index(papers, current_year):
    """
    Ha-index 계산 함수
    """
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

def career_years_h_index_by_average_citations_per_year(papers):
    count_year_dict = {}
    for citations,year in papers:
        if year not in count_year_dict:
            count_year_dict[year] = {'total_papers':0,'total_citations':0}
            
        count_year_dict[year]['total_papers'] +=1
        count_year_dict[year]['total_citations'] += citations
        result = {}
    for year,values in count_year_dict.items():
        total_papers = values['total_papers']
        total_citations = values['total_citations']
        average_citations = total_citations / total_papers if total_papers > 0 else 0
        result[year] = {'average_citations':average_citations}  

    sorted_years = sorted(result.items(),key=lambda x: x[1]['average_citations'],reverse=True)
    h_index = 0  # 初始化H指数
    for idx, (year, values) in enumerate(sorted_years):
        if values['average_citations'] >= (idx + 1):
            h_index = idx + 1
        else:
            break
    return h_index

def Timed_h_index(papers, start_year, end_year):
    count_year_dict = {}
    filtered_papers = [(citations, year) for citations, year in papers if start_year <= year <= end_year]
    for citations,year in filtered_papers:
        if year not in count_year_dict:
            count_year_dict[year] = {'total_citations':0}
        count_year_dict[year]['total_citations'] += citations
    for year,values in count_year_dict.items():
        total_citations = values['total_citations']
    sorted_years = sorted(count_year_dict.items(),key=lambda x: x[1]['total_citations'],reverse=True)
    print(sorted_years)
    h_index = 0  # 初始化H指数
    for idx, (year, values) in enumerate(sorted_years):
        if values['total_citations'] >= (idx + 1):
            h_index = idx + 1
        else:
            break
    return h_index

# def career_year_h_index(papers, current_year):
#     """
#     Career year h-index 계산 함수
#     """
#     avg_citations = sorted(
#         (c / (current_year - y + 1) for c, y in papers if current_year - y + 1 > 0),
#         reverse=True
#     )
#     return sum(1 for i, ac in enumerate(avg_citations) if ac >= i + 1)

# def timed_h_index(papers, current_year, t):
#     """
#     Timed h-index (최근 t년간 논문만 고려)
#     """
#     recent_papers = [(c, y) for c, y in papers if (current_year - y) < t]
#     citations = sorted((c for c, _ in recent_papers), reverse=True)
#     return sum(1 for i, c in enumerate(citations) if c >= i + 1)

# 전역 변수 정의
max_quality = None
max_combined_recency = None

def expert_index(papers, current_year, a=50, b=50):
    """
    전문가 지수 (Expert Index) 계산 함수.
    """
    if max_quality is None or max_combined_recency is None:
        raise ValueError("Error: preprocess_expert_index() must be called before expert_index().")

    if max_quality == 0 or max_combined_recency == 0:
        raise ValueError("Error: max_quality or max_combined_recency is zero. Check preprocessing.")

    recency_score, quality_score = rec_qual(papers, current_year)
    paper_count = len(papers)
    combined_recency = paper_count * recency_score

    norm_quality = quality_score / max_quality if max_quality else 0
    norm_recency = combined_recency / max_combined_recency if max_combined_recency else 0

    return (norm_quality * a) + (norm_recency * b)

def rec_qual(papers, current_year):
    """
    최신성(Recency) 및 품질(Quality) 점수 계산
    """
    recency_score = 0
    quality_score = 0
    
    for c, y in papers:
        recency = max(1 - (current_year - 3 - y) * 0.1, 0.1)
        recency_score += max(recency, 1)
        quality_score += math.log(c * math.exp((y - current_year) / 10) + 1)
    
    return recency_score, quality_score

def preprocess_expert_index(all_papers, current_year):
    """
    전체 연구자의 논문을 기반으로 전문가 지수를 정규화하기 위한 품질 및 최신성 최댓값을 미리 계산.
    """
    global max_quality, max_combined_recency

    if not all_papers:
        raise ValueError("Error: No papers found. Ensure that the dataset is correctly loaded.")

    max_quality = max(
        (math.log(c * math.exp((y - current_year) / 10) + 1) for c, y in all_papers),
        default=0
    )
    max_combined_recency = max(
        (len(all_papers) * max(1 - (current_year - 3 - y) * 0.1, 0.1) for _, y in all_papers),
        default=0
    )

    if max_quality == 0 or max_combined_recency == 0:
        raise ValueError("Error: max_quality or max_combined_recency is zero. Ensure proper preprocessing.")

    print(f"✅ Preprocessed max_quality: {max_quality}")
    print(f"✅ Preprocessed max_combined_recency: {max_combined_recency}")
