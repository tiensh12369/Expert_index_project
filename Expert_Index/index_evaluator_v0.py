import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr  # Import correlation function
import math

# Load authors data
def load_authors_data(authors_file):
    columns = [
        "validation_status", "author.author_id", "affiliation", "email_domain", "total_citations", "author_id"
    ]
    authors_data = []
    try:
        with open(authors_file, "r", encoding="cp1252") as f:
            for line in f:
                split_data = line.strip().split('|')
                if len(split_data) == 6:
                    authors_data.append(split_data)
                else:
                    # For authors without all fields, pad missing fields with None
                    authors_data.append(split_data + [None] * (6 - len(split_data)))
    except UnicodeDecodeError:
        print("Failed to open authors file with cp1252 encoding.")
    return pd.DataFrame(authors_data, columns=columns)

# Load author publication data
def load_author_publications(data_dir, author_id, current_year=2013):
    if author_id is None:
        return pd.DataFrame(columns=["citations", "year"])
    
    author_file = os.path.join(data_dir, f"{author_id}_.dat")
    if not os.path.exists(author_file):
        print(f"File {author_file} does not exist.")
        return pd.DataFrame(columns=["citations", "year"])
    publications = []
    try:
        with open(author_file, 'r', encoding='cp1252') as f:
            for line in f:
                citations, year = line.strip().split('|')
                if int(year) >= 1973 and int(year) <= current_year:
                    publications.append((int(citations), int(year)))
    except UnicodeDecodeError:
        print(f"Failed to open file {author_file} with the given encoding.")
    except FileNotFoundError:
        print(f"Publication data for author {author_id} not found.")
    return pd.DataFrame(publications, columns=["citations", "year"])

# Calculate final expert index using new formula
def calculate_index(dt, pYears, citations):
    recency_score = 0
    quality_score = 0
    
    for i in range(len(pYears)):
        
        recency = max(round((1 - (int(dt) - 3 - pYears[i]) * 0.1), 2), 0.1)
        
        if recency > 1:
            recency_score += 1
        else:
            recency_score += recency
        
        quality_score += math.log(int(citations[i]) * math.exp((int(pYears[i]) - dt) / 10) + 1)
    
    return recency_score, quality_score

# Calculate final expert index
def calculate_expert_index(author_data, data_dir, a, b):
    expert_indices = []
    max_quality_score = 0
    max_combined_recency = 0
    max_paper_count = 0
    intermediate_results = []
    
    # First pass to determine max values
    # Second pass to calculate expert indices
    for author in author_data.itertuples():
        if author.author_id is None:
            continue
        publications = load_author_publications(data_dir, author.author_id, current_year=2013)
        if publications.empty:
            continue
        
        # Calculate recency_score, quality_score, and activity
        recency_score, quality_score = calculate_index(2013, publications["year"].tolist(), publications["citations"].tolist())
        paper_count = len(publications)
        combined_recency = recency_score* paper_count
        
        # Update max values
        max_quality_score = max(max_quality_score, quality_score)
        max_combined_recency = max(max_combined_recency, combined_recency)
        max_paper_count = max(max_paper_count, paper_count)
        
        # Store intermediate results
        intermediate_results.append((author.author_id, quality_score, combined_recency, paper_count))

    for name, quality_score, combined_recency, paper_count in intermediate_results:
        # Normalize quality_score and combined_recency
        norm_quality_score = quality_score / max_quality_score if max_quality_score > 0 else 0
        norm_combined_recency = combined_recency / max_combined_recency if max_combined_recency > 0 else 0

        # Calculate final score
        score = (norm_quality_score * a) + (norm_combined_recency * b)
        
        expert_indices.append((name, score))
        
        recency_score, quality_score = calculate_index(2013, publications["year"].tolist(), publications["citations"].tolist())
        max_quality_score = max(max_quality_score, quality_score)
        max_combined_recency = max(max_combined_recency, recency_score* len(publications))
        max_paper_count = max(max_paper_count, len(publications))
    
    return pd.DataFrame(expert_indices, columns=["author_id", "expert_index"])

# Experiment evaluation: Comparing proposed index with existing indices
def evaluate_expert_indices(authors_data, data_dir):
    # Calculate h-index, g-index, and i10-index for comparison
    comparison_data = []
    for author in authors_data.itertuples():
        if author.author_id is None:
            continue
        publications = load_author_publications(data_dir, author.author_id)
        if publications.empty:
            continue
        
        # Calculate h-index
        sorted_citations = sorted(publications["citations"], reverse=True)
        h_index = sum(1 for i, c in enumerate(sorted_citations) if c >= i + 1)
        
        # Calculate g-index
        cumulative_citations = np.cumsum(sorted_citations)
        g_index = sum(1 for i, c in enumerate(cumulative_citations) if c >= (i + 1) ** 2)
        
        # Calculate i10-index
        i10_index = sum(1 for c in sorted_citations if c >= 10)
        
        comparison_data.append((author.author_id, h_index, g_index, i10_index))
    
    comparison_df = pd.DataFrame(comparison_data, columns=["author_id", "h_index", "g_index", "i10_index"])
    return comparison_df

# Find authors with significant rank changes between indices
def find_significant_rank_changes(top_10_h_index, top_10_g_index, top_10_i10_index, top_10_expert_index):
    all_authors = set(top_10_h_index['author_id']).union(set(top_10_g_index['author_id'])).union(set(top_10_i10_index['author_id'])).union(set(top_10_expert_index['author_id']))
    rank_changes = []
    for author in all_authors:
        ranks = {
            'h_index_rank': top_10_h_index[top_10_h_index['author_id'] == author].index.tolist(),
            'g_index_rank': top_10_g_index[top_10_g_index['author_id'] == author].index.tolist(),
            'i10_index_rank': top_10_i10_index[top_10_i10_index['author_id'] == author].index.tolist(),
            'expert_index_rank': top_10_expert_index[top_10_expert_index['author_id'] == author].index.tolist(),
        }
        # Replace empty lists with None for authors not in the top 10 of a given index
        for key in ranks:
            if not ranks[key]:
                ranks[key] = [None]
        rank_changes.append({
            'author_id': author,
            'h_index_rank': ranks['h_index_rank'][0] + 1 if ranks['h_index_rank'][0] is not None else None,
            'g_index_rank': ranks['g_index_rank'][0] + 1 if ranks['g_index_rank'][0] is not None else None,
            'i10_index_rank': ranks['i10_index_rank'][0] + 1 if ranks['i10_index_rank'][0] is not None else None,
            'expert_index_rank': ranks['expert_index_rank'][0] + 1 if ranks['expert_index_rank'][0] is not None else None
        })
    rank_changes_df = pd.DataFrame(rank_changes)
    rank_changes_df['rank_diff'] = rank_changes_df[['h_index_rank', 'g_index_rank', 'i10_index_rank', 'expert_index_rank']].apply(lambda x: max(x) - min(x) if None not in x.values else None, axis=1)
    significant_changes_df = rank_changes_df[rank_changes_df['rank_diff'] >= 3]  # Set a threshold of 3 or more positions for significant rank change
    print("Authors with Significant Rank Changes:")
    print(significant_changes_df)
    significant_changes_df.to_csv('significant_rank_changes.csv', index=False)
    return significant_changes_df

# Plotting results for analysis
def plot_expert_indices(expert_index_data, comparison_data):
    # Merge data
    merged_data = pd.merge(expert_index_data, comparison_data, on="author_id")
    
    # Calculate correlation coefficients
    h_corr, _ = pearsonr(merged_data["h_index"], merged_data["expert_index"])
    g_corr, _ = pearsonr(merged_data["g_index"], merged_data["expert_index"])
    i10_corr, _ = pearsonr(merged_data["i10_index"], merged_data["expert_index"])
    
    # Plot scatter plots to compare indices with correlation coefficients
    plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 3, 1)
    sns.scatterplot(x="h_index", y="expert_index", data=merged_data, alpha=0.5, hue="h_index", palette="viridis")
    plt.xlabel("h-index")
    plt.ylabel("Proposed Expert Index")
    plt.title(f"h-index vs Proposed Expert Index (Corr: {h_corr:.2f})")
    
    plt.subplot(1, 3, 2)
    sns.scatterplot(x="g_index", y="expert_index", data=merged_data, alpha=0.5, hue="g_index", palette="viridis")
    plt.xlabel("g-index")
    plt.ylabel("Proposed Expert Index")
    plt.title(f"g-index vs Proposed Expert Index (Corr: {g_corr:.2f})")
    
    plt.subplot(1, 3, 3)
    sns.scatterplot(x="i10_index", y="expert_index", data=merged_data, alpha=0.5, hue="i10_index", palette="viridis")
    plt.xlabel("i10-index")
    plt.ylabel("Proposed Expert Index")
    plt.title(f"i10-index vs Proposed Expert Index (Corr: {i10_corr:.2f})")
    
    plt.tight_layout()
    plt.show()
    
# Plotting results for analysis
def plot_expert_indices2(expert_index_data, comparison_data, expert_threshold=20, h_threshold=100):
    # Merge data
    merged_data = pd.merge(expert_index_data, comparison_data, on="author_id")
    
    # Filter data based on thresholds
    filtered_data = merged_data[(merged_data['expert_index'] <= expert_threshold) & (merged_data['h_index'] <= h_threshold)]
    
    # Calculate correlation coefficients
    h_corr, _ = pearsonr(filtered_data["h_index"], filtered_data["expert_index"])
    g_corr, _ = pearsonr(filtered_data["g_index"], filtered_data["expert_index"])
    i10_corr, _ = pearsonr(filtered_data["i10_index"], filtered_data["expert_index"])
    
    # Plot scatter plots to compare indices with correlation coefficients
    plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 3, 1)
    sns.scatterplot(x="h_index", y="expert_index", data=filtered_data, alpha=0.5, hue="h_index", palette="viridis")
    plt.xlabel("h-index")
    plt.ylabel("Proposed Expert Index")
    plt.title(f"h-index vs Proposed Expert Index (Corr: {h_corr:.2f})")
    
    plt.subplot(1, 3, 2)
    sns.scatterplot(x="g_index", y="expert_index", data=filtered_data, alpha=0.5, hue="g_index", palette="viridis")
    plt.xlabel("g-index")
    plt.ylabel("Proposed Expert Index")
    plt.title(f"g-index vs Proposed Expert Index (Corr: {g_corr:.2f})")
    
    plt.subplot(1, 3, 3)
    sns.scatterplot(x="i10_index", y="expert_index", data=filtered_data, alpha=0.5, hue="i10_index", palette="viridis")
    plt.xlabel("i10-index")
    plt.ylabel("Proposed Expert Index")
    plt.title(f"i10-index vs Proposed Expert Index (Corr: {i10_corr:.2f})")
    
    plt.tight_layout()
    plt.show()
  
# Time-based performance analysis
def plot_time_based_performance(author_data, data_dir, selected_authors=None):
    time_series_data = []
    for author in author_data.itertuples():
        if author.author_id is None:
            continue
        publications = load_author_publications(data_dir, author.author_id, current_year=2013)
        if publications.empty:
            continue

        # Calculate yearly expert index
        for year in range(1973, 2014):
            yearly_publications = publications[publications["year"] == year]
            if not yearly_publications.empty:
                recency_score, quality_score = calculate_index(year, yearly_publications["year"].tolist(), yearly_publications["citations"].tolist())
                paper_count = len(yearly_publications)
                combined_recency = recency_score * paper_count
                time_series_data.append((author.author_id, year, recency_score, quality_score, combined_recency))

    time_series_df = pd.DataFrame(time_series_data, columns=["author_id", "year", "recency_score", "quality_score", "combined_recency"])

    # Plot the time series for selected authors
    selected_authors = selected_authors if selected_authors is not None else time_series_df["author_id"].unique()[:5]  # Plot for the first 5 authors as an example
    plt.figure(figsize=(12, 8))
    for author in selected_authors:
        author_data = time_series_df[time_series_df["author_id"] == author]
        plt.plot(author_data["year"], author_data["combined_recency"], label=author)

    plt.xlabel("Year")
    plt.ylabel("Combined Recency Score")
    plt.title("Yearly Performance Analysis for Selected Authors")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def load_or_save_expert_index_data(authors_data, data_dir, expert_index_filename, comparison_index_filename, a, b):
    # Check if expert index results file already exists
    if os.path.exists(expert_index_filename):
        expert_index_data = pd.read_csv(expert_index_filename)
        print(f"Loaded existing expert index results from {expert_index_filename}.")
    else:
        # Calculate expert index for all authors
        expert_index_data = calculate_expert_index(authors_data, data_dir, a, b)
        
        # Sort by expert index in descending order and save results
        expert_index_data = expert_index_data.sort_values(by="expert_index", ascending=False)
        expert_index_data.to_csv(expert_index_filename, index=False)
        print(f"Expert index calculation complete. Results saved to {expert_index_filename}.")

    # Check if comparison index results file already exists
    if os.path.exists(comparison_index_filename):
        comparison_data = pd.read_csv(comparison_index_filename)
        print(f"Loaded existing expert index results from {comparison_index_filename}.")
    else:
        # Evaluate expert indices
        comparison_data = evaluate_expert_indices(authors_data, data_dir)
        comparison_data.to_csv(comparison_index_filename, index=False)
        print(f"Comparison indices calculation complete. Results saved to {comparison_index_filename}.")

    return expert_index_data, comparison_data

def compare_and_plot_top_authors(comparison_data, expert_index_data, data_dir):
    # Create a table to compare top 10 authors based on different indices
    top_100_h_index = comparison_data.sort_values(by='h_index', ascending=False)
    top_100_g_index = comparison_data.sort_values(by='g_index', ascending=False)
    top_100_i10_index = comparison_data.sort_values(by='i10_index', ascending=False)
    top_100_expert_index = expert_index_data.sort_values(by='expert_index', ascending=False)

    # Combine all top 10 lists into a single DataFrame
    rank_comparison_df = pd.DataFrame({
        'Rank': range(1, 78859),
        'Top 100 by h-index': top_100_h_index['author_id'].tolist(),
        'Top 100 by g-index': top_100_g_index['author_id'].tolist(),
        'Top 100 by i10-index': top_100_i10_index['author_id'].tolist(),
        'Top 100 by Expert Index': top_100_expert_index['author_id'].tolist()
    })

    print("Rank Comparison Table:")
    print(rank_comparison_df)
    rank_comparison_df.to_csv('rank_comparison_table.csv', index=False)

    top_experts = expert_index_data.head(20)  # Select top 10 authors by expert index
    merged_top_data = pd.merge(top_experts, comparison_data, on="author_id")

    # Add paper_count and total_citations for each author
    author_paper_counts = []
    for author in top_experts['author_id']:
        publications = load_author_publications(data_dir, author, current_year=2013)
        paper_count = len(publications)
        total_citations = publications['citations'].sum() if not publications.empty else 0
        author_paper_counts.append({'author_id': author, 'paper_count': paper_count, 'total_citations': total_citations})

    paper_count_df = pd.DataFrame(author_paper_counts)

    # Merge paper count and total_citations data
    merged_top_data = pd.merge(merged_top_data, paper_count_df, on='author_id')

    # Plot comparison of top experts with other indices
    plt.figure(figsize=(18, 8))

    plt.subplot(2, 3, 1)
    sns.barplot(x="author_id", y="h_index", data=merged_top_data, palette="viridis")
    plt.xlabel("Author ID")
    plt.ylabel("h-index")
    plt.title("Top Experts by Proposed Index vs h-index")
    plt.xticks(rotation=45)

    plt.subplot(2, 3, 2)
    sns.barplot(x="author_id", y="g_index", data=merged_top_data, palette="viridis")
    plt.xlabel("Author ID")
    plt.ylabel("g-index")
    plt.title("Top Experts by Proposed Index vs g-index")
    plt.xticks(rotation=45)

    plt.subplot(2, 3, 3)
    sns.barplot(x="author_id", y="i10_index", data=merged_top_data, palette="viridis")
    plt.xlabel("Author ID")
    plt.ylabel("i10-index")
    plt.title("Top Experts by Proposed Index vs i10-index")
    plt.xticks(rotation=45)

    # Plot number of papers and total citations for top experts
    plt.subplot(2, 3, 4)
    sns.barplot(x="author_id", y="paper_count", data=merged_top_data, palette="viridis")
    plt.xlabel("Author ID")
    plt.ylabel("Number of Papers")
    plt.title("Top Experts by Proposed Index vs Number of Papers")
    plt.xticks(rotation=45)
    plt.xlabel("Author ID")
    plt.ylabel("Number of Papers")
    plt.title("Top Experts by Proposed Index vs Number of Papers")
    plt.xticks(rotation=45)

    plt.subplot(2, 3, 5)
    sns.barplot(x="author_id", y="total_citations", data=merged_top_data, palette="viridis")
    plt.xlabel("Author ID")
    plt.ylabel("Total Citations")
    plt.title("Top Experts by Proposed Index vs Total Citations")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

    plt.tight_layout()
    plt.show()
    
    return top_experts

# Plot yearly citations for top authors
def plot_yearly_citations(top_experts, data_dir):
    yearly_data = []
    for author in top_experts["author_id"].tolist():
        publications = load_author_publications(data_dir, author, current_year=2013)
        if publications.empty:
            continue
        yearly_counts = publications.groupby("year").agg({"citations": "sum"})
        yearly_counts["author_id"] = author
        yearly_data.append(yearly_counts)

    yearly_df = pd.concat(yearly_data).reset_index()
    yearly_df.to_csv('yearly_df_citations.csv', index=False)
    plt.figure(figsize=(20, 6))
    sns.lineplot(x="year", y="citations", hue="author_id", data=yearly_df, marker="o")
    plt.xlabel("Year")
    plt.ylabel("Total Citations")
    plt.title("Yearly Citations for Top 20 Authors")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Plot yearly paper count for top authors
def plot_yearly_paper_count(top_experts, data_dir):
    yearly_data = []
    for author in top_experts["author_id"].tolist():
        publications = load_author_publications(data_dir, author, current_year=2013)
        if publications.empty:
            continue
        yearly_counts = publications.groupby("year").size().reset_index(name="paper_count")
        yearly_counts["author_id"] = author
        yearly_data.append(yearly_counts)

    yearly_df = pd.concat(yearly_data).reset_index(drop=True)
    yearly_df.to_csv('yearly_df_paper.csv', index=False)
    plt.figure(figsize=(20, 6))
    sns.lineplot(x="year", y="paper_count", hue="author_id", data=yearly_df, marker="o")
    plt.xlabel("Year")
    plt.ylabel("Total Paper Count")
    plt.title("Yearly Paper Count for Top 20 Authors")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()