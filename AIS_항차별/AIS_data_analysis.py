import gmplot
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.cluster import KMeans
import os.path
import math
import shutil
from tqdm import tqdm
from sklearn.metrics import silhouette_samples, silhouette_score

"""
코드를 사용하시기에 앞서 필요한 부분을 제외한 코드는 주석 처리 하여 사용하시면 됩니다.
주석 처리 방법은 주석 처리 하고 싶은 라인을 드래그 한 후
Ctrl + / 버튼을 이용해 주석 처리를 할 수 있고,
역으로 주석을 풀 때에도 같은 버튼을 이용해 주석을 풀 수 있습니다.
"""

"""
수정 가능한 파트
"""

#분석 파일명
file_path = '강원해양호_항적'

file = file_path

#모델 입력 값
google_map_flag = 0 # 구글맵에 항적표기 실행 여부. 1: 실행, 0: 실행 x
k_means_flag = 1 # k-means clustering 실행 여부. 1: 실행, 0: 실행 x
n_clusters = 4 # k(원하는 군집, 본 문제에서는 operation mode)의 개수를 설정

"""
수정 가능한 파트
"""

"""
시작: 데이터의 전처리를 진행하는 파트
입력값 : 원본파일.csv
출력값 : time_원본파일명.npy
"""

#if문이 끝나면 나오는 파일 명을 확인한 후 파일이 있으면 if문 내부를 패스하고, 파일이 없다면 if문 내부를 돈다
if os.path.isfile("./Output/time_{}.npy".format(file)) == False:
    
    """
    시작: Raw 데이터의 아웃라이어 및 불필요 항목 제거
    """
    
    # 파일을 불러들임, 다른 파일을 사용할 경우 해당 파일명을 입력하면 됨
    df = pd.read_csv("./Input" + file + ".csv", sep=',', skiprows=2, encoding='cp949')

    # 원래 데이터에 한글로 되어있어 인식되지 않는 칼럼 명을 영어로 변경함
    df.columns = ['MMSI', 'date', 'longi', 'latit', 'SOG', 'COG', 'heading', 'voyage']

    # drop이라는 함수를 이용해 사용하지 않는 'MMSI', 'COG', 'Heading'열을 제거함
    df = df.drop(['MMSI', 'COG', 'heading'], axis=1)
    
    # 원하는 조건에 맞는 데이터만 가져옴(이상치 제거). 이상치 제거를 위해서는 각 데이터의 값을 보고 원하는 조건을 입력하면 됨
    ###수정 가능한 파트###
    df = df[(df["longi"] < 38) & (df['SOG'] <= 30)]
    ###수정 가능한 파트###
    
    print("종료: Raw 데이터의 아웃라이어 및 불필요 항목 제거\n")
    
    """
    종료: Raw 데이터의 아웃라이어 및 불필요 항목 제거
    """
    
    """
    시작 : 날짜별 데이터로 저장하는 파트
    """
    
    df = df.reset_index(drop=True)
    
    #데이터의 date안에 있는 string 자료형으로 구성된 날짜 정보를 datetime이라는 날짜 정보를 쉽게 다룰 수 있는 자료형으로 변환
    df['date'] = pd.to_datetime(df['date'])

    ##SOG(속도)의 분포를 시각화 하는 파트###
    # fig, ax = plt.subplots()
    # sns.distplot(df['SOG'], hist=True, kde=True, bins=[0,5,10,15,20,30], ax=ax, hist_kws={'edgecolor' : 'gray'})
    # ax.set_xlim(0,30)
    # ax.set_xticks([0,5,10,15,20,30])
    # plt.xlabel("SOG (knot)")
    # plt.show()
    ##SOG(속도)의 분포를 시각화 하는 파트###
    
    ###불러온 데이터를 날짜별로 분류하는 파트###
    voyage_data = {}
    data_size = df.shape[0]
    voyage = 0
    
    for i in tqdm(range(data_size), desc='날짜별 분류 진행율'):
        if voyage == df['voyage'][i]:
            voyage_data["vayage_{}".format(df['voyage'][j])].append(df.loc[i].values.tolist())
            k=1
        else:
            j = i
            voyage_data["vayage_{}".format(df['voyage'][j])] = []
            voyage_data["vayage_{}".format(df['voyage'][j])].append(df.loc[i].values.tolist())
            voyage = df['voyage'][i]
    ###불러온 데이터를 날짜별로 분류하는 파트###
    
    #자료형이 dictionary 형태이기 때문에 dictionary를 객체 그대로 저장할 수 있는 pickle이라는 모듈을 이용해 저장
    with open('./Output/voyage_{}.pickle'.format(file),'wb') as fw:
        pickle.dump(voyage_data, fw)
                    
    print("종료 : 날짜별 데이터로 저장하는 파트\n")
    
    """
    종료 : 날짜별 데이터로 저장하는 파트
    """
    
    """
    시작: 날짜마다 속력 구간별 시간을 분단위 데이터로 저장하는 파트
    """
    
    time_data = np.zeros(shape=(len(voyage_data),5))

    ###time data 생성###
    for i, key in tqdm(enumerate(voyage_data), desc="시간 데이터 생성 진행율"):
        for j in range(len(voyage_data[key])):
            if j > 0:
                if voyage_data[key][j][0].day != voyage_data[key][j-1][0].day or \
                    voyage_data[key][j][0].hour != voyage_data[key][j-1][0].hour or \
                        voyage_data[key][j][0].minute != voyage_data[key][j-1][0].minute or \
                            voyage_data[key][j][0].second != voyage_data[key][j-1][0].second:
                            time_diff = voyage_data[key][j][0] - voyage_data[key][j-1][0]
                            time_diff = time_diff.seconds/60
                            if voyage_data[key][j][3] >= 0 and voyage_data[key][j][3] < 5:
                                time_data[i][0] += time_diff
                            elif voyage_data[key][j][3] >= 5 and voyage_data[key][j][3] < 10:
                                time_data[i][1] += time_diff
                            elif voyage_data[key][j][3] >= 10 and voyage_data[key][j][3] < 15:
                                time_data[i][2] += time_diff
                            elif voyage_data[key][j][3] >= 15 and voyage_data[key][j][3] < 20:
                                time_data[i][3] += time_diff
                            else:
                                time_data[i][4] += time_diff
    ###time data 생성###

    #데이터 저장
    np.save("./Output/time_{}".format(file), time_data)
    
    print("종료: 날짜마다 속력 구간별 시간을 분단위 데이터로 저장하는 파트\n")
    
    """
    종료: 날짜마다 속력 구간별 시간을 분단위 데이터로 저장하는 파트
    """

"""
종료: 데이터의 전처리를 진행하는 파트
"""

"""
시작: 처리된 데이터를 K-means clustering algorithm을 이용해 군집화 하는 파트
입력데이터: time_원본파일명.npy
출력데이터: None 
"""

if k_means_flag:
    
    #데이터 불러오기
    time_data = np.load("./Output/time_{}.npy".format(file), allow_pickle=True)

    #kmeans 모델 선언
    model = KMeans(n_clusters, n_init=10) #n_clusters = 군집의 수

    #K-means Clustering 진행
    model.fit(time_data)

    #K(군집의 중심)의 위치
    centers = model.cluster_centers_

    #데이터의 군집 번호 확인
    # print(model.labels_)
    labels = list(model.labels_)

    num_labels = set(labels)
    num_labels = list(num_labels)

    #각 군집별 데이터의 개수 확인
    print("data number of cluster")
    print("".join("label {} : {}\n".format(i, labels.count(i)) for i in num_labels))

    ###군집별 데이터를 시각화를 통해 확인하는 파트###
    bar_color = ['b', 'g', 'r', 'm', 'y', 'c', 'k', 'lime']
    col_num = math.ceil(n_clusters/2)
    plot_num = int("{}21".format(col_num))

    plt.figure(figsize=(9, 4.6*col_num))
    bar_x = ["0~5","5~10","10~15","15~20","20~30"]
    
    for i in range(len(centers)):
        plt.subplot(plot_num+i)
        plt.bar(bar_x, centers[i], color='{}'.format(bar_color[i]))
        plt.title('Operation Mode {} (n={})'.format(i+1, labels.count(i)), fontsize=10)
        plt.xlabel("(knot)")
        plt.ylabel("(minute)")


    ###군집별 데이터를 시각화를 통해 확인하는 파트###

    """
    date 파일을 불러와서 구글 맵 이미지로 저장하는 파트
    """

     #google_map이 1이면 아래 코드를 실행하고 0이면 실행하지 않음
    if google_map_flag:
        
        #데이터 불러오기
        with open('./Output/voyage_{}.pickle'.format(file),'rb') as fr:
            data = pickle.load(fr)

        #이미지 저장 경로
        if not os.path.exists("heatmap_{}".format(file)):
            os.makedirs("heatmap_{}".format(file))
        else:
            shutil.rmtree("heatmap_{}".format(file))
            os.makedirs("heatmap_{}".format(file))
            
        ###구글 맵 이미지 저장###
        for i, key in enumerate(data):
            longitude = []
            latitude = []
            for j in range(len(data[key])):
                longitude.append(data[key][j][1])
                latitude.append(data[key][j][2])
            
            mean_longitude = sum(longitude)/len(longitude)
            mean_latitude = sum(latitude)/len(latitude)
            
            gps_data = pd.DataFrame(list(zip(longitude, latitude)), columns=['LONGITUDE', 'LATITUDE'])
            
            gmap = gmplot.GoogleMapPlotter(mean_longitude, mean_latitude, 13)
            gmap.heatmap(gps_data['LONGITUDE'], gps_data['LATITUDE'])

            gmap.draw(os.path.join("heatmap_{}".format(file), "Mode{}_voyage_{}.html".format(labels[i],\
                 data[key][0][4])))
        ###구글 맵 이미지 저장###
    
    """
    date 파일을 불러와서 구글 맵 이미지를 띄우는 파트
    """
print('---------------------------------------------------------------------')
print('silhouette score: ',silhouette_score(time_data, model.labels_))

plt.show()

"""
종료: 처리된 데이터를 K-means clustering algorithm을 이용해 군집화 하는 파트
"""