import os
import pandas as pd
import matplotlib.pyplot as plt
from index_evaluator import load_authors_data, evaluate_all_with_expert_index
from data_utils import load_author_publications
from index_metrics import (
    h_index, g_index, i10_index, ha_index, ar_index,
    timed_h_index, contemporary_h_index, trend_h_index,
    career_years_h_index_by_average_citations_per_year,
)
from viz_utils import *

# 설정
authors_file = "./gsc_data/authors.all"
data_dir = "./gsc_data/DATA/"
output_file = "all_index_results.csv"
current_year = 2013

start_year = 1970
end_year = 2013

authors_data = load_authors_data(authors_file)

for year in range(start_year, end_year + 1):
    expert_by_year_file = f"index_by_{year}.csv"
    if not os.path.exists(expert_by_year_file):
        print(f"📆 전체 author 연도별 지수 계산 중...{expert_by_year_file}")
        index_df, max_q, max_r = evaluate_all_with_expert_index(authors_data, data_dir, year)
        print(max_q, max_r)
        index_df.to_csv(expert_by_year_file, index=False)
        print(f"{year}년도 계산 완료")
    else:
        index_df = pd.read_csv(expert_by_year_file)
        print(f"{year}년도 로드 완료")
        pass
        # df_expert_by_year = pd.read_csv(expert_by_year_file)
        # print("📂 기존 expert_index_by_year.csv 로드 완료")
        
# 2. 컬럼 이름 축약 및 분석 대상 설정
index_df_short = index_df.rename(columns=INDEX_NAME_MAP)
index_columns = list(INDEX_NAME_MAP.values())

# 3. 분석 및 시각화 실행
print("📊 상관관계 분석 중...")
plot_index_scatter(index_df, "expert_index", "h_index")
plot_index_scatter(index_df, "expert_index", "g_index")
plot_index_scatter(index_df, "expert_index", "i10_index")
plot_index_scatter(index_df, "expert_index", "contemporary_h_index")
plot_index_scatter(index_df, "expert_index", "career_year_h_index_by_publications")
plot_index_scatter(index_df, "expert_index", "career_year_h_index_by_publications_year_citations")
plot_index_scatter(index_df, "expert_index", "career_years_h_index_by_average_citations_per_year")
plot_index_scatter(index_df, "expert_index", "timed_h_index_5")
plot_index_scatter(index_df, "expert_index", "timed_h_index_10")
plot_index_scatter(index_df, "expert_index", "ha_index")

corr_pearson = plot_index_correlation_heatmap(index_df, method='pearson')
print(corr_pearson)
corr_spearman = plot_index_correlation_heatmap(index_df, method='spearman')
print(corr_spearman)

print("🔝 Top-K 비교 및 겹침 분석 중...")
print_top_k_authors(index_df_short, k=1000, index_columns=index_columns)
overlap_matrix = compute_top_k_overlap_matrix(index_df_short, index_columns, k=1000)
# print(overlap_matrix)
plot_top_k_overlap_heatmap(overlap_matrix)

print("🧍 독립 저자 분석 중...")
unique_authors_by_index = find_index_specific_top_k_authors(index_df_short, index_columns, k=20)
for idx, authors in unique_authors_by_index.items():
    print(f"🔹 {idx} 지수에만 Top-K로 포함된 저자 {len(authors)}명: {authors}")

print("🔢 Top-K 확장에 따른 고유 저자 수 시각화 중...")
k_range = list(range(10, 110, 10))
author_counts_by_k = plot_unique_author_counts_by_k(index_df_short, index_columns, k_values=k_range)
print(author_counts_by_k)