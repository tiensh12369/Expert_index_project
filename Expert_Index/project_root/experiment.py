import pandas as pd
import matplotlib.pyplot as plt
from data_loader import load_authors, load_papers
from index_calculator import calculate_h_index, calculate_g_index, calculate_i10_index
from index_comparator import compare_indices
import seaborn as sns

def run_experiment(author_file, data_folder, top_n):
    """
    여러 지수를 계산하고 비교하는 실험을 실행하는 함수.
    :param author_file: 저자 정보 파일 경로
    :param data_folder: 논문 데이터 폴더 경로
    :param top_n: 비교할 상위 저자 수
    """
    authors_df = load_authors(author_file)
    rankings = {"h_index": [], "g_index": [], "i10_index": []}
    
    for _, row in authors_df.iterrows():
        papers = load_papers(data_folder, row['ID'])
        rankings["h_index"].append((row['ID'], calculate_h_index(papers)))
        rankings["g_index"].append((row['ID'], calculate_g_index(papers)))
        rankings["i10_index"].append((row['ID'], calculate_i10_index(papers)))
    
    for key in rankings:
        rankings[key] = sorted(rankings[key], key=lambda x: x[1], reverse=True)
    
    comparison_matrix = compare_indices(rankings, top_n)
    df = pd.DataFrame(comparison_matrix)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(df, annot=True, cmap='Blues')
    plt.title("Index Overlap Comparison")
    plt.show()
    
    return df
