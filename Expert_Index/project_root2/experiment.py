import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import pearsonr
from data_loader import load_authors, load_papers
from index_calculator import (
    h_index, g_index, i10_index, ar_index, contemporary_h_index, ha_index,
    career_year_h_index, timed_h_index, expert_index, preprocess_expert_index
)
from index_comparator import compare_indices

def analyze_author(author_id, papers_cache, current_year, periods=[3, 5, 10]):
    """
    특정 저자의 연구 경력, 논문 수, 인용 수를 분석하는 함수.
    """
    papers = papers_cache.get(author_id, [])
    if not papers:
        return {
            "Career Length": 0,
            "Total Papers": 0,
            "Total Citations": 0,
            "Avg Citations per Paper": 0,
            **{f"Papers last {p} years": 0 for p in periods},
            **{f"Citations last {p} years": 0 for p in periods},
            **{f"Avg Citations last {p} years": 0 for p in periods}
        }
    
    first_year = min(y for _, y in papers)
    career_length = current_year - first_year + 1
    total_papers = len(papers)
    total_citations = sum(c for c, _ in papers)
    avg_citations = total_citations / total_papers if total_papers > 0 else 0
    
    period_stats = {}
    for p in periods:
        recent_papers = [c for c, y in papers if y >= current_year - p]
        total_recent_citations = sum(recent_papers)
        avg_recent_citations = total_recent_citations / len(recent_papers) if recent_papers else 0
        period_stats[f"Papers last {p} years"] = len(recent_papers)
        period_stats[f"Citations last {p} years"] = total_recent_citations
        period_stats[f"Avg Citations last {p} years"] = avg_recent_citations
    
    author_stats = {
        "Career Length": career_length,
        "Total Papers": total_papers,
        "Total Citations": total_citations,
        "Avg Citations per Paper": avg_citations,
        **period_stats
    }
    
    return author_stats

def analyze_top_authors(rankings, papers_cache, top_n, current_year):
    """
    상위 저자들의 연구 경향을 분석하는 함수.
    """
    top_authors = {}

    for index_name, ranking in rankings.items():
        top_authors[index_name] = [author_id for author_id, _ in ranking[:top_n]]

    analysis_results = {}
    for index_name, author_ids in top_authors.items():
        stats = {key: 0 for key in analyze_author(author_ids[0], papers_cache, current_year).keys()}

        for author_id in author_ids:
            author_stats = analyze_author(author_id, papers_cache, current_year)
            for key in stats:
                stats[key] += author_stats[key]

        analysis_results[index_name] = {key: stats[key] / top_n for key in stats}

    df_analysis = pd.DataFrame(analysis_results).T
    df_analysis.to_csv("./results/top_authors_analysis.csv")
    print("\nTop Authors Analysis saved to ./results/top_authors_analysis.csv")

    return df_analysis


def preprocess_global_expert_index(author_files, data_folder, current_year):
    """
    전체 연구자의 품질 및 최신성 최댓값을 구하는 전처리 과정.
    """
    global max_quality, max_combined_recency

    authors_df = load_authors(author_files)
    papers_cache = {}
    all_papers = []
    
    for _, row in authors_df.iterrows():
        author_id = row['ID']
        if author_id not in papers_cache:
            papers_cache[author_id] = load_papers(data_folder, author_id)
        all_papers.extend(papers_cache[author_id])

    if not all_papers:
        raise ValueError("No papers found. Ensure that the dataset is correctly loaded.")

    preprocess_expert_index(all_papers, current_year)


def run_experiment(author_file, data_folder, top_n, current_year, t=5, gamma=4, delta=1, a=50, b=50):
    """
    여러 지수를 계산하고 비교하는 실험을 실행하는 함수.
    """
    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)  # 결과 저장 디렉토리 생성

    authors_df = load_authors(author_file)
    papers_cache = {}
    rankings = {
        "h_index": [], "g_index": [], "i10_index": [], "ar_index": [], "contemporary_h_index": [],
        "ha_index": [], "career_year_h_index": [], "timed_h_index": [], "expert_index": []
    }
    
    print("Preprocessing expert index...")  # 🔹 전처리 시작 메시지 추가
    preprocess_global_expert_index(author_file, data_folder, current_year)
    print("Preprocessing complete. Running experiment...")  # 🔹 전처리 완료 메시지 추가

    for _, row in authors_df.iterrows():
        author_id = row['ID']
        if author_id not in papers_cache:
            papers_cache[author_id] = load_papers(data_folder, author_id)
        papers = papers_cache[author_id]

        rankings["h_index"].append((author_id, h_index(papers)))
        rankings["g_index"].append((author_id, g_index(papers)))
        rankings["i10_index"].append((author_id, i10_index(papers)))
        rankings["ar_index"].append((author_id, ar_index(papers, current_year)))
        rankings["contemporary_h_index"].append((author_id, contemporary_h_index(papers, current_year, gamma, delta)))
        rankings["ha_index"].append((author_id, ha_index(papers, current_year)))
        rankings["career_year_h_index"].append((author_id, career_year_h_index(papers, current_year)))
        rankings["timed_h_index"].append((author_id, timed_h_index(papers, current_year, t)))
        rankings["expert_index"].append((author_id, expert_index(papers, current_year, a, b)))

    for key in rankings:
        rankings[key] = sorted(rankings[key], key=lambda x: x[1], reverse=True)

    # 🔹 compare_indices() 호출 전 rankings 변환
    rankings_ids = {key: [author_id for author_id, _ in rankings[key]] for key in rankings}
    comparison_matrix = compare_indices(rankings_ids, top_n)

    df_comparison = pd.DataFrame(comparison_matrix)
    df_comparison.to_csv(f"{results_dir}/comparison_matrix.csv", index=False)
    print("\nComparison Matrix saved to ./results/comparison_matrix.csv")

    # 🔹 상위 저자 분석 수행
    df_analysis = analyze_top_authors(rankings, papers_cache, top_n, current_year)
    print(df_analysis)

    # 🔹 분석 결과 저장 경로 통일 및 index=False 추가
    df_analysis.to_csv(f"{results_dir}/index_statistics.csv", index=False)
    print("\nIndex statistics saved to ./results/index_statistics.csv")

    # 🔹 지수 간 상관 관계 분석 추가
    df_rankings = pd.DataFrame({key: [val[1] for val in rankings[key]] for key in rankings})
    correlation_matrix = df_rankings.corr()
    correlation_matrix.to_csv(f"{results_dir}/correlation_matrix.csv", index=False)
    print("\nCorrelation Matrix saved to ./results/correlation_matrix.csv")

    # 🔹 상관 관계 히트맵 시각화
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="Blues")
    plt.title("Index Correlation Matrix")
    plt.show()

    return df_analysis
