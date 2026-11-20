def count_duplicates(list1, list2):
    """
    두 지수의 상위 저자 리스트 간 중복 개수를 계산하는 함수.
    :param list1: 첫 번째 지수의 상위 저자 리스트
    :param list2: 두 번째 지수의 상위 저자 리스트
    :return: 중복된 요소 개수
    """
    return len(set(list1) & set(list2))

def compare_indices(rankings, author_num):
    """
    여러 지수의 상위 저자 리스트 간 중복성을 비교하는 함수.
    :param rankings: 여러 지수의 정렬된 저자 ID 리스트
    :param author_num: 비교할 저자 수
    :return: 중복 비율 매트릭스
    """
    index_names = list(rankings.keys())
    matrix = {name: {other: 0 for other in index_names} for name in index_names}
    
    for i, key1 in enumerate(index_names):
        for j, key2 in enumerate(index_names):
            if i <= j:
                matrix[key1][key2] = count_duplicates(rankings[key1][:author_num], rankings[key2][:author_num]) / author_num
                matrix[key2][key1] = matrix[key1][key2]
    
    return matrix
