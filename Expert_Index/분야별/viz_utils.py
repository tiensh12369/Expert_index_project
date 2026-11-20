import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from data_utils import load_author_publications
from IPython.display import display
from scipy.stats import zscore

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# ✅ 컬럼 축약 매핑 사전
INDEX_NAME_MAP = {
    'h_index': 'h',
    'g_index': 'g',
    'i10_index': 'i10',
    'contemporary_h_index': 'hc',
    'ar_index': 'ar',
    'hpd_index': 'hpd',
    # 'trend_h_index': 'ht',
    'career_year_h_index_by_publications': 'cy_pub',
    'career_year_h_index_by_publications_year_citations': 'cy_cit',
    'career_years_h_index_by_average_citations_per_year': 'cy_avg',
    'timed_h_index_5': 'h_t5',
    'timed_h_index_10': 'h_t10',
    'ha_index': 'ha',
    'expert_index': 'expert'
}

def get_styled_index_map(index_columns):

    # ✅ 컬러팔레트 (색상 최대 10개까지 시각적으로 잘 구분됨)
    color_palette = plt.get_cmap("tab10").colors

    # ✅ 마커 리스트 (논문용/발표용에서 시각적으로 명확한 형태)
    marker_list = ["o", "s", "D", "^", "v", "*", "X", "<", ">"]

    # ✅ 선 스타일 반복
    linestyle_list = ["-", "--", "-."]

    index_styles = {}

    for i, col in enumerate(index_columns):
        index_styles[col] = {
            "color": color_palette[i % len(color_palette)],
            "marker": marker_list[i % len(marker_list)],
            "linestyle": linestyle_list[i % len(linestyle_list)],
        }

    return index_styles


# ✅ 컬럼 이름 축약 적용
def shorten_index_names(df):
    return df.rename(columns=INDEX_NAME_MAP)

def get_top_k_authors(index_df, k=20, index_cols=None):
    """
    각 지수별로 Top-K author_id 리스트를 반환하는 공통 함수

    Parameters:
    - index_df: author_id + 지수 컬럼이 포함된 DataFrame
    - k: Top-K 수
    - index_cols: 사용할 지수 컬럼명 리스트 (None이면 모든 수치형 컬럼 자동 선택)

    Returns:
    - top_k_dict: {지수명: [author_id, ...]} 형태의 딕셔너리
    """
    short_df = shorten_index_names(index_df)
    
    if index_cols is None:
        index_cols = [col for col in short_df.columns if col != "author_id" and pd.api.types.is_numeric_dtype(short_df[col])]
    
    top_k_dict = {}
    for idx in index_cols:
        top_k_dict[idx] = (
            short_df[["author_id", idx]]
            .sort_values(by=idx, ascending=False)
            .head(k)["author_id"]
            .tolist()
        )
    return top_k_dict


# ✅ 상관관계 히트맵 시각화
def plot_index_correlation_heatmap(df, method='pearson'):
    short_df = shorten_index_names(df)

    # 분석 대상 컬럼만 선택 (숫자형만, author_id 등 제외)
    exclude_cols = {'author_id', 'recency_score', 'quality_score', 'paper_count'}
    index_cols = [col for col in short_df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(short_df[col])]

    # 상관계수 계산 및 시각화
    corr = short_df[index_cols].corr(method=method)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title(f"{method.capitalize()} Correlation between Indices")
    plt.tight_layout()
    plt.show()
    
    return corr

def print_top_k_authors(df, k=10, index_columns=None, id_col="author_id"):
    print(f"🔝 Top-{k} Authors by Each Index:\\n")
    for col in index_columns:
        print(f"📌 {col}")
        top_authors = df[[id_col, col]].sort_values(by=col, ascending=False).head(k)
        print(top_authors.to_string(index=False))
        print("-" * 40)

# ✅ Top-K 저자 간 겹침 수 교차표 계산
def compute_top_k_overlap_matrix(df, index_columns, k=10, id_col="author_id"):
    overlap_matrix = pd.DataFrame(index=index_columns, columns=index_columns)

    # 각 지수별 Top-K 저자 set 생성
    top_k_dict = get_top_k_authors(df, k, index_columns)
    top_k_sets = {col: set(ids) for col, ids in top_k_dict.items()}

    # 교차 비교 (겹치는 수 계산)
    for col1 in index_columns:
        for col2 in index_columns:
            overlap = len(top_k_sets[col1].intersection(top_k_sets[col2]))
            ratio = overlap / k
            overlap_matrix.loc[col1, col2] = ratio

    return overlap_matrix.round(1)

# ✅ 히트맵 시각화
def plot_top_k_overlap_heatmap(overlap_matrix):
    plt.figure(figsize=(10, 8))
    sns.heatmap(overlap_matrix, annot=True, fmt="d", cmap="YlGnBu", square=True)
    plt.title("Top-K Author Overlap Between Indices")
    plt.tight_layout()
    plt.show()

def find_index_specific_top_k_authors(df, index_columns, k=10, id_col="author_id"):
    top_k_dict = get_top_k_authors(df, k, index_columns)
    top_k_sets = {col: set(ids) for col, ids in top_k_dict.items()}

    results = {}
    for col in index_columns:
        others = set().union(*[v for k2, v in top_k_sets.items() if k2 != col])
        unique_authors = top_k_sets[col] - others
        results[col] = unique_authors
    return results

def plot_rank_shift(df, index_columns, base_index, top_n=15, id_col="author_id"):
    rank_df = df[[id_col] + index_columns].copy()
    for col in index_columns:
        rank_df[f"{col}_rank"] = rank_df[col].rank(ascending=False, method='min')

    base_rank = rank_df[f"{base_index}_rank"]
    shift_data = {}
    for col in index_columns:
        if col == base_index:
            continue
        shift = (rank_df[f"{col}_rank"] - base_rank).abs()
        top_authors = shift.sort_values(ascending=False).head(top_n).index
        shift_data[col] = shift.loc[top_authors]

    shift_df = pd.DataFrame(shift_data)
    shift_df.plot(kind='bar', figsize=(12, 6), title=f"Top-{top_n} Rank Shift Compared to '{base_index}'")
    plt.ylabel("Absolute Rank Difference")
    plt.xlabel("Author Index")
    plt.tight_layout()
    plt.show()

def plot_unique_author_counts_by_k(df, index_columns, k_values, id_col="author_id"):
    data = []

    # ✅ Top-K 유니크 저자 수 계산
    for k in k_values:
        top_k_dict = get_top_k_authors(df, k, index_columns)
        top_k_sets = {col: set(ids) for col, ids in top_k_dict.items()}

        for col in index_columns:
            others = set().union(*[v for k2, v in top_k_sets.items() if k2 != col])
            unique_authors = top_k_sets[col] - others
            data.append({
                "index": col,
                "k": k,
                "unique_authors": len(unique_authors)
            })

    result_df = pd.DataFrame(data)

    # ✅ 스타일 맵 가져오기
    index_styles = get_styled_index_map(index_columns)

    # ✅ 시각화
    plt.figure(figsize=(12, 6))
    for col in index_columns:
        subset = result_df[result_df["index"] == col]
        style = index_styles[col]
        plt.plot(
            subset["k"],
            subset["unique_authors"],
            label=col,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"]
        )

    plt.title("Top-K Only Authors per Index")
    plt.ylabel("Number of Unique Authors")
    plt.xlabel("Top-K")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    return result_df


def plot_index_clustermap(overlap_matrix, k):
    # 유사도 → 거리 행렬로 변환
    dist_matrix = 1 - overlap_matrix.astype(float) / k

    # Clustermap 시각화
    sns.clustermap(dist_matrix, cmap="Blues", annot=True, fmt=".2f")
    plt.title("Index Clustering by Top-K Overlap")
    plt.show()

# 시각화 예시: ha-index vs h-index
def plot_index_scatter(df, x_col, y_col):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=x_col, y=y_col, data=df)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"{y_col} vs {x_col}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
def analyze_unique_authors_characteristics(unique_authors, data_dir, current_year=2013):
    all_years = []
    all_citations = []
    paper_counts = []
    
    for author_id in unique_authors:
        df = load_author_publications(data_dir, author_id, current_year)
        if df.empty:
            continue
        all_years.extend(df["year"])
        all_citations.extend(df["citations"])
        paper_counts.append(len(df))

    # 1. 연도 분포 시각화
    plt.figure(figsize=(10, 4))
    sns.histplot(all_years, bins=range(min(all_years), current_year+1), kde=False)
    plt.title("Publication Year Distribution of Unique Authors")
    plt.xlabel("Year")
    plt.ylabel("Number of Papers")
    plt.tight_layout()
    plt.show()

    # 2. 인용수 분포
    plt.figure(figsize=(10, 4))
    sns.histplot(all_citations, bins=20, kde=False)
    plt.title("Citation Distribution of Unique Authors")
    plt.xlabel("Citations per Paper")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    # 3. 평균 논문 수와 인용수 출력
    print(f"🧾 저자 수: {len(paper_counts)}")
    print(f"📚 평균 논문 수: {sum(paper_counts)/len(paper_counts):.2f}")
    print(f"📈 평균 인용 수: {sum(all_citations)/len(all_citations):.2f}")

def analyze_extreme_rank_shift(df, index_columns, base_index='expert', top_n=5, id_col='author_id'):
    # 모든 지수에 대한 순위 계산
    rank_df = df[[id_col] + index_columns].copy()
    for col in index_columns:
        rank_df[f"{col}_rank"] = rank_df[col].rank(ascending=False, method='min')

    base_rank = rank_df[f"{base_index}_rank"]

    # 지수별로 기준 지수와의 랭킹 차이 계산
    for col in index_columns:
        if col != base_index:
            rank_df[f"{col}_rank_diff"] = (rank_df[f"{col}_rank"] - base_rank).abs()

    # 최대 변화가 큰 저자 추출
    rank_df["max_rank_diff"] = rank_df[[f"{col}_rank_diff" for col in index_columns if col != base_index]].max(axis=1)
    top_authors = rank_df.sort_values(by="max_rank_diff", ascending=False).head(top_n)

    # 결과 테이블 출력
    display_cols = [id_col] + index_columns + [f"{col}_rank" for col in index_columns]
    print(f"🧠 기준 지수: {base_index} 기준, 랭킹 변화가 큰 Top-{top_n} 저자")
    
    display(top_authors[display_cols])
    
    return top_authors[display_cols]

def plot_author_index_and_activity_time_series(author_id, index_df, data_dir, selected_columns=None, normalize_zscore=True):
    """
    특정 저자의 지수 시계열과 논문 수/인용 수를 시각화하는 함수
    """
    short_df = shorten_index_names(index_df)

    if selected_columns is None:
        exclude_cols = {"author_id", "year", "quality_score", "recency_score"}
        selected_columns = [
            col for col in short_df.columns
            if col not in exclude_cols and short_df[col].dtype != "object"
        ]

    styled_map = get_styled_index_map(index_columns=selected_columns)
    plot_df = short_df.copy()

    if normalize_zscore:
        plot_df[selected_columns] = plot_df[selected_columns].apply(zscore, nan_policy='omit')

    # 📈 지수 시계열 그래프
    plt.figure(figsize=(12, 6))
    for col in selected_columns:
        if col in plot_df.columns:
            style = styled_map.get(col, {})
            plt.plot(
                plot_df["year"], plot_df[col],
                label=col,
                marker=style.get("marker", "o"),
                linestyle=style.get("linestyle", "-"),
                color=style.get("color", "black")
            )

    plt.title(f"Index Time Series for {author_id} (Z-Score Normalized)")
    plt.xlabel("Year")
    plt.ylabel("Z-Score" if normalize_zscore else "Raw Score")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 📊 논문/인용수 시각화
    df_papers = load_author_publications(data_dir, author_id, current_year=2013)
    df_papers["year"] = df_papers["year"].astype(int)
    
    paper_stats = df_papers.groupby("year")["citations"].agg(["count", "sum"]).reset_index()
    paper_stats.rename(columns={"count": "papers", "sum": "citations"}, inplace=True)
    
    full_years = pd.DataFrame({"year": range(1970, 2013 + 1)})
    paper_stats = full_years.merge(paper_stats, on="year", how="left").fillna(0)
    
    paper_stats["papers"] = paper_stats["papers"].astype(int)
    paper_stats["citations"] = paper_stats["citations"].astype(int)

    if not paper_stats.empty:
        paper_stats.set_index("year")[["papers"]].plot(
            kind="bar", figsize=(12, 4), title=f"Number of Papers per Year: {author_id}", color="skyblue"
        )
        plt.ylabel("Papers")
        plt.grid(axis="y")
        plt.tight_layout()
        plt.show()

        paper_stats.set_index("year")[["citations"]].plot(
            kind="bar", figsize=(12, 4), title=f"Number of Citations per Year: {author_id}", color="salmon"
        )
        plt.ylabel("Citations")
        plt.grid(axis="y")
        plt.tight_layout()
        plt.show()
        
        return plot_df, paper_stats
    else:
        print("⚠️ 논문 데이터가 없어 활동 시각화를 건너뜁니다.")

