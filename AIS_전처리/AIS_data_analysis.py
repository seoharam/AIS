import pandas as pd
from datetime import datetime

def get_datetime_obejct(str_object):
    date_format = "%Y-%m-%d %H:%M:%S"
    date_object = datetime.strptime(str_object, date_format)
    
    return date_object

# csv 파일로 dataframe 생성
dataframe_AIS = pd.read_csv("./input/1. 한우리호_항적.csv", engine='python', encoding='cp949',  skiprows=2)
df_AIS = dataframe_AIS[["MMSI", "일시", "SOG", "항차번호"]]

# dict 생성
result_dict = {}

# 한줄씩 dataframe을 불러옴
for index, row in df_AIS.iterrows():
    
    # tuple안에 들어갈 요소
    date_object = get_datetime_obejct(row["일시"])
    sog_trans = 0.5144 * row["SOG"]
    
    # 해당 딕셔너리에 키 값이 있는지 확인
    if row['항차번호'] not in result_dict:
        result_dict[row['항차번호']] = [(date_object, sog_trans, 0)]

    # 있다면, 기준점의 값과의 시간차이를 계산해서 기존 리스트에 튜플을 append
    else:
        deviation = date_object - result_dict[row['항차번호']][0][0]
        second_deviation = deviation.seconds
        result_dict[row['항차번호']].append((date_object, sog_trans, second_deviation))