import os
import pandas as pd
import math

from index_metrics import calculate_all_indices
from data_utils import load_author_publications  # ✅ 공통 모듈에서 함수 사용


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


def compute_expert_components(papers, current_year):
    quality_score = 0
    recency_score = 0
    for c, y in papers:
        if y == 0:
            return 0, 0
        
        rec = max(round((1 - 0.1 * (current_year - 3 - y)), 2), 0.1)
        
        if rec > 1:
            recency_score += 1
        else:
            recency_score += rec
            
        quality_score += math.log(c * math.exp((y - current_year) / 10) + 1)
    return quality_score, recency_score

def evaluate_all_with_expert_index(authors_data, data_dir, current_year=2013, a=50, b=50):
    results = []

    for _, row in authors_data.iterrows():
        author_id = row["author_id"]
        df = load_author_publications(data_dir, author_id, current_year)

        papers = list(map(tuple, df.values))
        if not papers:
            continue

        indices = calculate_all_indices(papers, current_year)
        q, r = compute_expert_components(papers, current_year)

        result = {"author_id": author_id}
        result.update(indices)
        result["quality_score"] = q
        result["recency_score"] = r
        results.append(result)

    # 정규화
    max_q = max(r["quality_score"] for r in results) or 1
    max_r = max(r["recency_score"] for r in results) or 1

    for r in results:
        q_norm = r["quality_score"] / max_q
        r_norm = r["recency_score"] / max_r
        r["expert_index"] = a * q_norm + b * r_norm  # ✅ 논문 정의와 일치

    df = pd.DataFrame(results)

    # 랭킹 계산 (높을수록 좋은 순위)
    # for col in df.columns:
    #     if col not in {"author_id"} and pd.api.types.is_numeric_dtype(df[col]):
    #         df[f"{col}_rank"] = df[col].rank(method="min", ascending=False)

    return df, max_q, max_r
