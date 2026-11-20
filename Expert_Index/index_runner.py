import os
from index_evaluator import load_authors_data, evaluate_all_with_expert_index
from viz_utils import plot_index_correlation_heatmap
import pandas as pd

# 경로 설정
authors_file = "./gsc_data/authors.all"
data_dir = "./gsc_data/DATA/"
output_file = "all_index_results.csv"

# 데이터 로드
authors_data = load_authors_data(authors_file)

if not os.path.exists(output_file):
    index_df = evaluate_all_with_expert_index(authors_data, data_dir, current_year=2013)
    index_df.to_csv(output_file, index=False)
else:
    index_df = pd.read_csv(output_file)
    
from viz_utils import plot_index_scatter

# 예시 시각화 실행
plot_index_scatter(index_df, "h_index", "ha_index")
plot_index_correlation_heatmap(index_df, method='pearson')
plot_index_correlation_heatmap(index_df, method='spearman')


from viz_utils import print_top_k_authors, INDEX_NAME_MAP

# 사용한 지수 컬럼 (축약 후)
short_cols = list(INDEX_NAME_MAP.values())

# 결과 데이터에서 축약 컬럼 적용
index_df_short = index_df.rename(columns=INDEX_NAME_MAP)

# Top-10 저자 출력
print_top_k_authors(index_df_short, k=10, index_columns=short_cols)

from viz_utils import compute_top_k_overlap_matrix, plot_top_k_overlap_heatmap, INDEX_NAME_MAP

# 지수 이름 축약
index_df_short = index_df.rename(columns=INDEX_NAME_MAP)
index_columns = list(INDEX_NAME_MAP.values())

# Top-K 교차 분석
overlap_matrix = compute_top_k_overlap_matrix(index_df_short, index_columns, k=20)

# 결과 확인 및 시각화
print(overlap_matrix)
plot_top_k_overlap_heatmap(overlap_matrix)

from viz_utils import (
    find_index_specific_top_k_authors,
    plot_rank_shift,
    INDEX_NAME_MAP
)

index_df_short = index_df.rename(columns=INDEX_NAME_MAP)
index_columns = list(INDEX_NAME_MAP.values())

# 1. 독립 Top-K 저자 찾기
unique_authors_by_index = find_index_specific_top_k_authors(index_df_short, index_columns, k=20)
for idx, authors in unique_authors_by_index.items():
    print(f"🔹 {idx} 지수에만 Top-K로 포함된 저자 {len(authors)}명: {authors}")

# 2. 랭킹 변화 시각화 (expert 기준)
plot_rank_shift(index_df_short, index_columns, base_index='expert', top_n=20)

from viz_utils import plot_unique_author_counts_by_k, INDEX_NAME_MAP

index_df_short = index_df.rename(columns=INDEX_NAME_MAP)
index_columns = list(INDEX_NAME_MAP.values())
k_range = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

plot_unique_author_counts_by_k(index_df_short, index_columns, k_values=k_range)

from viz_utils import analyze_unique_authors_characteristics

unique_authors_by_index = find_index_specific_top_k_authors(index_df_short, index_columns, k=30)
cy_avg_unique_authors = unique_authors_by_index["expert"]

analyze_unique_authors_characteristics(cy_avg_unique_authors, data_dir, current_year=2013)

from viz_utils import plot_index_clustermap

plot_index_clustermap(overlap_matrix, k=30)

from viz_utils import INDEX_NAME_MAP, analyze_extreme_rank_shift

index_df_short = index_df.rename(columns=INDEX_NAME_MAP)
index_columns = list(INDEX_NAME_MAP.values())

extreme_cases = analyze_extreme_rank_shift(index_df_short, index_columns, base_index='expert', top_n=5)
import os
import math
import pandas as pd

# 1. quality score 계산
def compute_quality_score(papers, current_year):
    return sum(math.log(c * math.exp((y - current_year) / 10) + 1) for c, y in papers)

# 2. recency score 계산
def compute_recency_score(papers, current_year):
    rec = 0
    for _, y in papers:
        val = max(round((1 - (current_year - 3 - y) * 0.1), 2), 0.1)
        rec += max(val, 1)
    return rec * len(papers)

# 3. 저자 논문 불러오기
def load_author_publications(data_dir, author_id, current_year):
    file_path = os.path.join(data_dir, f"{author_id}_.dat")
    if not os.path.exists(file_path):
        return []
    with open(file_path, encoding="cp1252") as f:
        lines = [line.strip().split("|") for line in f if "|" in line]
        return [(int(c), int(y)) for c, y in lines if int(y) <= current_year]

# 4. 전체 저자 연도별 expert 지수 계산
def evaluate_expert_by_year(authors_file, data_dir, start_year, end_year, a=50, b=50):
    with open(authors_file, encoding="cp1252") as f:
        author_ids = [line.strip().split("|")[-1] for line in f if "|" in line and line.strip().split("|")[-1]]

    all_records = []

    for year in range(start_year, end_year + 1):
        print(f"⏳ Processing year {year}...")

        year_scores = []
        for author_id in author_ids:
            papers = load_author_publications(data_dir, author_id, current_year=year)
            if not papers:
                continue
            q = compute_quality_score(papers, year)
            r = compute_recency_score(papers, year)
            year_scores.append((author_id, q, r))

        if not year_scores:
            continue

        max_q = max(q for _, q, _ in year_scores) or 1
        max_r = max(r for _, _, r in year_scores) or 1

        for author_id, q, r in year_scores:
            expert = a * (q / max_q) + b * (r / max_r)
            all_records.append({
                "author_id": author_id,
                "year": year,
                "quality_score": q,
                "recency_score": r,
                "expert_index": expert
            })

    return pd.DataFrame(all_records)

df_expert_series = evaluate_expert_by_year(
    authors_file="./gsc_data/authors.all",
    data_dir="./gsc_data/DATA/",
    start_year=1970,
    end_year=2013
)

# 결과 저장
df_expert_series.to_csv("expert_index_by_year.csv", index=False)


top_10_per_year = df_expert_series.groupby("year").apply(
    lambda x: x.nlargest(10, "expert_index")[["author_id", "expert_index"]]
)
print(top_10_per_year)

author = "ar-SwCsAAAAJ"
df_author = df_expert_series[df_expert_series.author_id == author]

import matplotlib.pyplot as plt
plt.plot(df_author["year"], df_author["expert_index"], marker="o")
plt.title(f"Expert Index Over Time: {author}")
plt.xlabel("Year")
plt.ylabel("Expert Index")
plt.grid(True)
plt.show()

df_expert_series.groupby("year")["expert_index"].mean().plot(
    marker="o", title="Average Expert Index Over Time"
)

import seaborn as sns
sns.boxplot(x="year", y="expert_index", data=df_expert_series)
plt.title("Expert Index Distribution by Year")
plt.grid(True)
plt.show()

def find_emerging_authors(df, year_target, k=20):
    top_sets = {}
    for year in sorted(df["year"].unique()):
        top_k = df[df["year"] == year].nlargest(k, "expert_index")["author_id"]
        top_sets[year] = set(top_k)

    # 2013 Top-K에서 이전 연도 Top-K에 없던 author
    previous = set().union(*[top_sets[y] for y in top_sets if y < year_target])
    current = top_sets[year_target]
    new_authors = current - previous
    return new_authors

newbies = find_emerging_authors(df_expert_series, year_target=2013, k=20)
df_newbies = df_expert_series[df_expert_series["author_id"].isin(newbies)]

import seaborn as sns
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_newbies, x="year", y="expert_index", hue="author_id", marker="o")
plt.title("Emerging Authors in 2013 (Top-K newcomers)")
plt.xlabel("Year")
plt.ylabel("Expert Index")
plt.grid(True)
plt.legend(title="Author ID", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()

# 2013년 Top-20 중, 이전 연도에는 없던 신진 연구자 추출
new_authors = find_emerging_authors(df_expert_series, year_target=2013, k=20)
print("🔍 신진 연구자 목록 (2013 기준):")
print(new_authors)

example_author = list(new_authors)[0]  # 첫 번째 저자 예시 선택
df_one = df_expert_series[df_expert_series.author_id == example_author]

# 시각화
import matplotlib.pyplot as plt
plt.plot(df_one["year"], df_one["expert_index"], marker="o", label="expert")
plt.plot(df_one["year"], df_one["quality_score"], marker="s", label="quality", linestyle="--")
plt.plot(df_one["year"], df_one["recency_score"], marker="^", label="recency", linestyle=":")
plt.title(f"Expert Components Over Time: {example_author}")
plt.xlabel("Year")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.show()

target = example_author  # 이전에 찾은 신진 저자
row = index_df_short[index_df_short["author_id"] == target]

# 점수와 순위 추출
score_cols = list(INDEX_NAME_MAP.values())
rank_cols = [f"{col}_rank" for col in score_cols]

summary = row[["author_id"] + score_cols + rank_cols].T
print(summary)

df_papers = load_author_publications(data_dir, example_author, current_year=2013)

# 연도별 논문 수
papers_per_year = df_papers.groupby("year").size()

# 연도별 인용 수 총합
citations_per_year = df_papers.groupby("year")["citations"].sum()

# 시각화
import matplotlib.pyplot as plt
fig, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(papers_per_year.index, papers_per_year.values, marker="o", label="Paper Count")
ax1.set_ylabel("Number of Papers")

ax2 = ax1.twinx()
ax2.plot(citations_per_year.index, citations_per_year.values, marker="s", label="Citations", color="orange")
ax2.set_ylabel("Total Citations")

plt.title(f"Paper & Citation Trend: {example_author}")
plt.grid(True)
fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
plt.tight_layout()
plt.show()
