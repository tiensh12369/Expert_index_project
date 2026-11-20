import os
import pandas as pd

def load_authors(file_path):
    """
    Google Scholar 데이터에서 저자 정보를 로드하는 함수.
    :param file_path: authors.all 파일 경로
    :return: 저자 ID 및 이름이 포함된 DataFrame
    """
    author_id_dict = {"Name": [], "ID": []}
    
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            fields = line.strip().split('|')
            if len(fields) >= 2:
                author_id_dict["Name"].append(fields[1])
                author_id_dict["ID"].append(fields[-1])
    
    return pd.DataFrame(author_id_dict)

def load_papers(folder_path, author_id):
    """
    특정 저자의 논문 데이터를 로드하는 함수.
    :param folder_path: 논문 데이터가 저장된 폴더 경로
    :param author_id: 특정 저자의 ID
    :return: 논문 리스트 [(인용 수, 출판 연도), ...]
    """
    file_path = os.path.join(folder_path, f"{author_id}_.dat")
    papers = []
    
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            for line in lines:
                fields = line.strip().split('|')
                if len(fields) >= 2:
                    citations = int(fields[0])
                    year = int(fields[1])
                    papers.append((citations, year))
    
    return papers
