import os
import pandas as pd
import math
from index_metrics import calculate_all_indices

def load_authors_data(authors_file):
    columns = [
        "validation_status", "author.author_id", "affiliation",
        "email_domain", "total_citations", "author_id"
    ]
    authors_data = []
    with open(authors_file, "r", encoding="cp1252") as f:
        for line in f:
            split_data = line.strip().split('|')
            if len(split_data) == 6:
                authors_data.append(split_data)
            else:
                authors_data.append(split_data + [None] * (6 - len(split_data)))
    return pd.DataFrame(authors_data, columns=columns)

def load_author_publications(data_dir, author_id, current_year=2013):
    if author_id is None:
        return pd.DataFrame(columns=["citations", "year"])
    
    author_file = os.path.join(data_dir, f"{author_id}_.dat")
    if not os.path.exists(author_file):
        return pd.DataFrame(columns=["citations", "year"])
    
    publications = []
    with open(author_file, 'r', encoding='cp1252') as f:
        for line in f:
            citations, year = line.strip().split('|')
            if int(year) <= current_year:
                publications.append((int(citations), int(year)))
    
    return pd.DataFrame(publications, columns=["citations", "year"])

# ⬇ 전문가 지수 구성 요소 계산
def compute_expert_components(papers, current_year):
    recency_score = 0
    quality_score = 0
    for c, y in papers:
        recency = max(round((1 - (current_year - 3 - y) * 0.1), 2), 0.1)
        recency_score += max(recency, 1)
        quality_score += math.log(c * math.exp((y - current_year) / 10) + 1)
    return recency_score, quality_score, len(papers)

# ✅ 전문가 지수 포함 전체 인덱스 평가 함수
def evaluate_all_with_expert_index(authors_data, data_dir, current_year=2013, a=50, b=50):
    temp = []

    # 1차 루프: 각 저자별 요소 계산
    for author in authors_data.itertuples():
        if author.author_id is None:
            continue
        pub_df = load_author_publications(data_dir, author.author_id, current_year)
        if pub_df.empty:
            continue
        papers = pub_df.to_records(index=False)
        rec, qual, count = compute_expert_components(papers, current_year)
        temp.append({
            "author_id": author.author_id,
            "recency_score": rec,
            "quality_score": qual,
            "paper_count": count,
            "combined_recency": rec * count,
            "raw_indices": calculate_all_indices(papers, current_year)
        })

    # 최대값 정규화
    max_quality = max(row["quality_score"] for row in temp)
    max_combined_recency = max(row["combined_recency"] for row in temp)

    results = []
    for row in temp:
        norm_qual = row["quality_score"] / max_quality if max_quality else 0
        norm_rec = row["combined_recency"] / max_combined_recency if max_combined_recency else 0
        expert_score = norm_qual * a + norm_rec * b

        record = {
            "author_id": row["author_id"],
            "expert_index": expert_score,
            "recency_score": row["recency_score"],
            "quality_score": row["quality_score"],
            "paper_count": row["paper_count"]
        }
        record.update(row["raw_indices"])
        results.append(record)

    return pd.DataFrame(results)
