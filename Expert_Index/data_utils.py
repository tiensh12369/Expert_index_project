# data_utils.py

import os
import pandas as pd

def load_author_publications(data_dir, author_id, current_year=None):
    """
    Load a single author's publication data as a DataFrame.
    Columns: [citations, year]
    Filters out future years if current_year is set.
    """
    file_path = os.path.join(data_dir, f"{author_id}_.dat")
    if not os.path.exists(file_path):
        return pd.DataFrame(columns=["citations", "year"])

    with open(file_path, encoding="cp1252") as f:
        lines = [line.strip().split("|") for line in f if "|" in line]
        records = [(int(c), int(y)) for c, y in lines if not current_year or int(y) <= current_year]
    return pd.DataFrame(records, columns=["citations", "year"])

def load_authors_data(file_path):
    """
    Load author metadata from a CSV file into a DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Author file not found: {file_path}")
    return pd.read_csv(file_path)

def load_index_df(file_path):
    """
    Load index dataframe (with scores/ranks) from CSV.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Index file not found: {file_path}")
    return pd.read_csv(file_path)

def load_author_index_time_series(author_id, index_by_year_dir, start_year=1970, end_year=2013):
    """
    연도별 저장된 index_by_YYYY.csv 파일에서 특정 author의 지수 시계열 정보를 가져오는 함수
    """
    records = []

    for year in range(start_year, end_year + 1):
        file_path = os.path.join(index_by_year_dir, f"index_by_{year}.csv")
        if not os.path.exists(file_path):
            continue

        df = pd.read_csv(file_path)
        author_row = df[df["author_id"] == author_id]
        if not author_row.empty:
            row_data = author_row.iloc[0].to_dict()
            row_data["year"] = year
            records.append(row_data)

    return pd.DataFrame(records)
