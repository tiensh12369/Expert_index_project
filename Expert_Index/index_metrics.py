import math
import numpy as np

def h_index(papers):
    citations = sorted([c for c, _ in papers], reverse=True)
    return sum(1 for i, c in enumerate(citations, 1) if c >= i)

def g_index(papers):
    citations = sorted([c for c, _ in papers], reverse=True)
    total = 0
    g = 0
    for i, c in enumerate(citations, 1):
        total += c
        if total >= i * i:
            g = i
        else:
            break
    return g

def i10_index(papers):
    return sum(1 for c, _ in papers if c >= 10)

def ar_index(papers, current_year):
    if not papers:
        return 0
    # 1. 인용수 기준 내림차순 정렬
    sorted_papers = sorted(papers, key=lambda x: x[0], reverse=True)
    # 2. h-index 계산
    h = h_index(papers)
    core = sorted_papers[:h]
    # 3. 감쇠 인용 합산
    ar_score = sum(c / (current_year - y + 1) for c, y in core)
    return ar_score ** 0.5

def ha_index(papers, current_year):
    avg_citations = [c / (current_year - y + 1) for c, y in papers if y <= current_year]
    avg_citations.sort(reverse=True)
    ha = 0
    for i, ac in enumerate(avg_citations, 1):
        if ac >= i:
            ha = i
        else:
            break
    return ha

def hpd_index(papers, current_year):
    if not papers:
        return 0
    cpdi_list = []
    for citations, pub_year in papers:
        age_in_decades = (current_year - pub_year + 1) / 10
        cpdi = citations / age_in_decades if age_in_decades > 0 else 0
        cpdi_list.append(cpdi)
    cpdi_list.sort(reverse=True)
    for i, cpdi in enumerate(cpdi_list, 1):
        if cpdi < i:
            return i - 1
    return len(cpdi_list)

def timed_h_index(papers, current_year, t=5):
    filtered = [(c, y) for c, y in papers if current_year - y < t]
    return h_index(filtered)

def contemporary_h_index(papers, current_year, gamma=4, delta=1):
    """
    Sidiropoulos et al. (2007)
    SC_i = gamma / (Y - Y_i + 1)^delta * C_i
    hC = max i s.t. SC_i >= i
    """
    if not papers:
        return 0
    sc_scores = []
    for c, y in papers:
        age = current_year - y + 1
        score = gamma / (age ** delta) * c if age > 0 else gamma * c
        sc_scores.append(score)

    sc_scores.sort(reverse=True)
    for i, sc in enumerate(sc_scores, 1):
        if sc < i:
            return i - 1
    return len(sc_scores)

def trend_h_index(papers, current_year, gamma=4, delta=1):
    trend_scores = []
    for c, y in papers:
        if y > current_year:
            continue
        score = sum(gamma / (current_year - y_ + delta) for y_ in [y]*c)
        trend_scores.append(score)
    trend_scores.sort(reverse=True)
    return sum(1 for i, ts in enumerate(trend_scores, 1) if ts >= i)

def career_year_h_index_by_publications(papers):
    """
    Mahbuba and Rousseau (2013):
    최대 h: h개의 연도가 각각 h개 이상의 논문을 가질 때
    """
    if not papers:
        return 0
    # 연도별 출판 수 계산
    year_pubs = {}
    for _, y in papers:
        if y not in year_pubs:
            year_pubs[y] = 0
        year_pubs[y] += 1

    pub_counts = sorted(year_pubs.values(), reverse=True)

    for i, count in enumerate(pub_counts, 1):
        if count < i:
            return i - 1
    return len(pub_counts)

def career_year_h_index_by_publications_year_citations(papers):
    """
    Mahbuba and Rousseau (2013):
    최대 h: h개의 연도가 각각 h회 이상의 총 인용 수를 가질 때
    """
    if not papers:
        return 0
    # 연도별 인용 수 계산
    year_citations = {}
    for c, y in papers:
        if y not in year_citations:
            year_citations[y] = 0
        year_citations[y] += c

    # 인용 수 기준 내림차순 정렬
    citation_counts = sorted(year_citations.values(), reverse=True)

    # h-style 조건 적용
    for i, total_cit in enumerate(citation_counts, 1):
        if total_cit < i:
            return i - 1
    return len(citation_counts)

def career_years_h_index_by_average_citations_per_year(papers):
    year_totals = {}
    for c, y in papers:
        if y not in year_totals:
            year_totals[y] = {'total_citations': 0, 'count': 0}
        year_totals[y]['total_citations'] += c
        year_totals[y]['count'] += 1

    avg_per_year = [v['total_citations'] / v['count'] for v in year_totals.values()]
    avg_per_year.sort(reverse=True)

    h = 0
    for i, avg in enumerate(avg_per_year, 1):
        if avg >= i:
            h = i
        else:
            break

    # 보간 적용
    if h < len(avg_per_year):
        ch = avg_per_year[h - 1]  # h번째 평균 인용수
        ch1 = avg_per_year[h]     # h+1번째 평균 인용수
        if ch != ch1:
            hint = ((h + 1) * ch - h * ch1) / (1 - ch1 + ch)
            return hint

    return float(h)

def twci(papers, current_year, t=5, gamma=1.0):

    """
    논문 목록에 대해 IV-T 지수를 계산하는 함수.

    Parameters:
    - papers: List[Tuple[int, int]]
        논문 목록. 각 항목은 (인용수, 출판연도)의 튜플.
    - T1: int
        시작 연도 (평가 구간의 하한)
    - T2: int
        종료 연도 (평가 구간의 상한이자 기준 시점)
    - gamma: float
        시간 감쇠 계수 (기본값은 1.5)

    Returns:
    - float
        계산된 TWCI 점수
    """
    T1 = current_year - t
    T2 = current_year
    
    twci_score = 0.0
    for citations, pub_year in papers:
        if T1 <= pub_year <= T2:
            age = T2 - pub_year + 1
            twci_score += np.log(citations + 1) / (age ** gamma)
    return twci_score
    

# def calculate_all_indices(papers, current_year):
#     return {
#         "h_index": h_index(papers),
#         "g_index": g_index(papers),
#         "i10_index": i10_index(papers),
#         "contemporary_h_index": contemporary_h_index(papers, current_year),
#         "ar_index": ar_index(papers, current_year),
#         "hpd_index": hpd_index(papers, current_year),
#         # "trend_h_index": trend_h_index(papers, current_year),
#         "career_year_h_index_by_publications": career_year_h_index_by_publications(papers),
#         "career_year_h_index_by_publications_year_citations": career_year_h_index_by_publications_year_citations(papers),
#         "career_years_h_index_by_average_citations_per_year": career_years_h_index_by_average_citations_per_year(papers),
#         "timed_h_index_5": timed_h_index(papers, current_year, t=5),
#         "timed_h_index_10": timed_h_index(papers, current_year, t=10),
#         "ha_index": ha_index(papers, current_year)
#     }
    
def calculate_all_indices(papers, current_year):
    return {
        "h_index": h_index(papers),
        "i10_index": i10_index(papers),
        "contemporary_h_index": contemporary_h_index(papers, current_year),
        "ar_index": ar_index(papers, current_year),
        "timed_h_index_5": timed_h_index(papers, current_year, t=5),
        "ha_index": ha_index(papers, current_year)
    }
    
# def evaluate_expert_by_year(authors_file, data_dir, start_year, end_year, a=50, b=50):
#     """
#     연도별로 전체 author의 expert 점수를 계산하여 시계열 DataFrame 생성
#     """
#     import pandas as pd
#     import math
#     from data_utils import load_author_publications

#     def compute_quality_score(papers, current_year):
#         return sum(math.log(c * math.exp((y - current_year) / 10) + 1) for c, y in papers)

#     def compute_recency_score(papers, current_year):
#         rec = 0
#         for _, y in papers:
#             val = max(round((1 - (current_year - 3 - y) * 0.1), 2), 0.1)
            
#             if val > 1:
#                 rec += 1
#             else:
#                 rec += rec
#         return rec * len(papers)

#     with open(authors_file, encoding="cp1252") as f:
#         author_ids = [line.strip().split("|")[-1] for line in f if "|" in line]

#     all_records = []

#     for year in range(start_year, end_year + 1):
#         year_scores = []
#         for author_id in author_ids:
#             df = load_author_publications(data_dir, author_id, current_year=year)
#             papers = list(zip(df["citations"], df["year"]))
#             if not papers:
#                 continue
#             q = compute_quality_score(papers, year)
#             r = compute_recency_score(papers, year)
#             year_scores.append((author_id, q, r))

#         max_q = max(q for _, q, _ in year_scores) or 1
#         max_r = max(r for _, _, r in year_scores) or 1

#         for author_id, q, r in year_scores:
#             expert = a * (q / max_q) + b * (r / max_r)
#             all_records.append({"author_id": author_id, "year": year, "quality": q, "recency": r, "expert": expert})

#     return pd.DataFrame(all_records)
