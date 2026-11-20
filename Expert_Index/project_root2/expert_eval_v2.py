import expert_index_lib as eil

authors_file = r"./gsc_data/authors.all"
data_dir = r"./gsc_data/DATA/"

expert_index_filename = f"expert_index_results.csv"
comparison_index_filename = f"comparison_index_results.csv"

# Load data
authors_data = eil.load_authors_data(authors_file)

expert_index_data, comparison_data = eil.load_or_save_expert_index_data(authors_data, data_dir, expert_index_filename, comparison_index_filename, a, b)

# Plot results
eil.plot_expert_indices(expert_index_data, comparison_data)
eil.plot_expert_indices2(expert_index_data, comparison_data)

# Create a table to compare top 10 authors based on different indices
top_experts = eil.compare_and_plot_top_authors(comparison_data, expert_index_data, data_dir)

# Plot time-based performance analysis for top 10 authors
# eil.plot_time_based_performance(authors_data, data_dir, selected_authors=top_experts["author_id"].tolist())

# Plot yearly citations for top 10 authors
eil.plot_yearly_citations(top_experts, data_dir)
eil.plot_yearly_paper_count(top_experts, data_dir)