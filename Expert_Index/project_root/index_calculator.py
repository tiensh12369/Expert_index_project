import math

def calculate_h_index(papers):
    """
    H-index 계산 함수
    :param papers: 논문 인용 횟수와 연도 리스트 [(citations, year), ...]
    :return: H-index
    """
    citations = sorted([c for c, _ in papers], reverse=True)
    h_index = sum(1 for i, c in enumerate(citations, 1) if c >= i)
    return h_index

def calculate_g_index(papers):
    """
    G-index 계산 함수
    :param papers: 논문 인용 횟수와 연도 리스트 [(citations, year), ...]
    :return: G-index
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

def calculate_i10_index(papers):
    """
    I10-index 계산 함수
    :param papers: 논문 인용 횟수와 연도 리스트 [(citations, year), ...]
    :return: I10-index
    """
    return sum(1 for c, _ in papers if c >= 10)
