import pandas as pd #판다스 import
import numpy as np #넘파이 import
import matplotlib.pyplot as plt #matplotlib import
import calendar # calendar import 
from statsmodels.tsa.seasonal import seasonal_decompose # 시계열 분석 패키지 import 
# Seasonal Decomposition
from tslearn.clustering import TimeSeriesKMeans # 시계열 클러스터링
from tslearn.preprocessing import TimeSeriesScalerMeanVariance # 시계열 스케일링
from yellowbrick.cluster.elbow import kelbow_visualizer
from datetime import datetime

def anlyUsekwh(data, info_costn, custinfo): # anlyUsekwh 함수 정의 (인자: data, info_costn, custinfo)
    #파라미터 날짜 범위가 아닌 조회된 데이터의 시작/종료 날짜 얻기
    get_start = data['datetime'].min() # get_start = data의 datetime 열의 최소값 (시작 날짜)
    get_end = data['datetime'].max() # get_end = data의 datetime 열의 최댓값 (종료 날짜)
        
    # start : 데이터가 0시부터 없는 경우도 고려 / end : 데이터가 23시까지 없는 경우도 고려  
    start = get_start.replace(hour=0, minute=0, second=0) 
    # start = 시작 날짜 데이터의 hour, minute, second를 0시 0분 0초로 바꾼 것
    # 시작 날짜가 0시 0분 0초가 아닐 경우 -> 0시 0분 0초로 변경
    end = get_end.replace(hour=23, minute=0, second=0)
    # end = 종료 날짜 데이터의 hour, minute, second를 23시 0분 0초로 바꾼 것
    # 종료 날짜기 23시 0시 0분이 아닐 경우 -> 23시 0분 0초로 변경

    # #####15분 단위 전력사용 데이터 처리#####
    # # 다계기인 경우, 계기 별로 데이터 분리, 결측치 처리    
    r = pd.date_range(start=start, end=end, freq = '1H', name='datetime')
    # r = 시작 날짜부터 종료 날짜를 1시간 주기로 나눈 datetimeindex를 datetime이라는 이름으로 지정함
    data_meter=pd.DataFrame(index=r)
    # data_meter = 인덱스가 r인 데이터 프레임 생성
    data_temp = data.copy(deep=True)
    # data_temp = data 복사본 생성 (deep copy-원본과 별개의 복사본 생성)
    data_temp = data_temp.astype({ 'usekWh' : 'float' , 'demandkW' : 'float'})
    # data_temp의 usekWh, demandkW 컬럼의 형식을 실수형으로 지정 (소수점이 있는 숫자 자료형)
    # Set 'datetime' as the index
    data_temp.set_index('datetime', inplace=True)
    # data_temp의 index를 datetime 컬럼으로 설정, inplace = True (기존 컬럼을 그대로 index로 설정)
    # Resample data to 1-hour frequency for each meter
    data_meter_cur =  data_temp.groupby('meterNo').resample('1H').agg({'usekWh': 'first'}).unstack('meterNo')
    # data_meter_cur = data_temp를 'meterNo'를 기준으로 그룹화 하고 1시간 단위로 usekWh의 첫번째 값을 추출한 후 그 결과를 meterNo별로 펼침
    data_meter_cur.columns = data_meter_cur.columns.droplevel()  # Remove the multi-index levels
    # data_meter_cur.columns에서 인덱스를 제거

    # 생성한 datetime index 기준으로 join
    data_meter = data_meter.join(data_meter_cur)
    # data_meter와 data_meter_cur를 join
    # 측정이 안 된 일자 혹은 시간 : Fill missing values with 0
    data_meter.fillna(0, inplace=True) 

   
    ############### data_meter에서 결측값을 0으로 채우기 -> 결측치를 0으로 채우는 이유? ###############
   
    data_meter['usekWh']=round(data_meter.sum(axis=1)).astype('int64')
    # data_meter의 usdkWh열 -> data_meter라는 DataFrame을 행별로 더하고 형식을 int64로 설정

    dailyEnergy=pd.DataFrame(data_meter['usekWh'].resample('24H').sum())
    #dailyEnerge = data_meter에서 usekWh열을 24시간 단위로 나눈 후 열 별로 합한 배열 

    #UsekWhStat : 요일별/월별 통계 
    data_day=pd.DataFrame(data_meter["usekWh"].groupby(data_meter.index.weekday).sum()).sort_values(by='usekWh', ascending=False)
    # data_day = data_meter의 usekWh열을 data_meter의 index의 weekday별로 그룹화 하여 합한 값을 usekWh 내림차순으로 정렬한 행렬
    data_month=pd.DataFrame(data_meter["usekWh"].groupby(data_meter.index.month).sum()).sort_values(by='usekWh', ascending=False)
    # data_month = data_meter의 usekWh열을 data_meter의 index의 month별로 그룹화 하여 합한 값을 usekWh 내림차순으로 정렬한 행렬

    usekWhStats={'FirstMonth' : data_month.iloc[0].name,
                             'SecondMonth' : data_month.iloc[1].name,
                             'LastMonth' : data_month.iloc[len(data_month)-1].name,
                             'FirstDay' : data_day.iloc[0].name,
                             'SecondDay' : data_day.iloc[1].name,
                             'LastDay' : data_day.iloc[len(data_day)-1].name}
    
    # usekWhStats 생성
    # FirstMonth : data_month에서 0번째 행의 name -> 첫번째 달 
    # SecondMonth : data_month에서 1번째 행의 name -> 두번째 달
    # LastMonth : data_month에서 마지막 행의 name -> 마지막 달
    # FirstDay : data_day에서 0번째 행의 name -> 첫번째 일 
    # SecondDay : data_day에서 1번째 행의 name -> 두번째 일 
    # LastDay : data_day에서 마지막 행의 name -> 마지막 일 


    #Anormaly : 가동중지 추정일
    result=seasonal_decompose(dailyEn rgy["usekWh"], model='additive')
    # seasonal_decompose : df

    ############### seasonal_decompose 패키지를 사용한 이유 ###############

    constance=1.5
    
    Q1 = np.percentile(result.resid.dropna(), 25)
    Q3 = np.percentile(result.resid.dropna(), 75)
    IQR = Q3 - Q1 
    IQR = IQR if IQR > 0 else -1*IQR
    lower = Q1 - constance*IQR
    higher = Q3 + constance*IQR

    # 잔차 이상치 분석 
    
    for i in result.resid.dropna().index:
        if result.resid.dropna().loc[i] < lower or  result.resid.dropna().loc[i] > higher:
            dailyEnergy.at[i,"Anormaly"]=1
            dailyEnergy.at[i, "usekWhAft"]=dailyEnergy.usekWh.mean()
        else:
            dailyEnergy.at[i,"Anormaly"]=0
            dailyEnergy.at[i, "usekWhAft"]=dailyEnergy.at[i, "usekWh"]
    
    # result.resid에서 na를 제거한 후의 index를 i라고 할때
    # 만약 result.resid에서 결측치를 제거한 값의 i가 lower보다 작거나, higher보다 크면 -> 잔차가 이상치라면 
    # dailyEnergy의 i번째 행의 Anormaly 열 값을 1로 지정
    # 그 후 dailyEnergy의 i번째 usekWhAft 열 값을 dailyEnergy의 usekWh의 평균값으로 지정

    ############### 이상치를 평균값으로 처리한 이유 ###############

    # 만약 result.resid에서 결측치를 제거한 값의 i가 Q1과 Q3 사이의 값이라면 -> 잔차가 이상치가 아니라면
    # dailyEnergy의 i번째 행의 Anormaly 열 값을 0으로 지정 
    # 그 후 dailyEnergy의 i번째 usekWhAft 열 값을 dailyEnergy의 i번째 행 usekWh 값을 그대로 사용

    for i in dailyEnergy[dailyEnergy["Anormaly"].isna()].index:
        dailyEnergy.at[i, "Anormaly"]=0
        dailyEnergy.at[i, "usekWhAft"]=dailyEnergy.at[i, "usekWh"]
        
    # dailyEnergy의 Anormaly값이 결측치인 행의 인덱스를 i라고 할 때
    # dailyEnergy의 i번째 행의 Anormaly열 값을 0으로 지정
    # dailyEnergy의 i번째 행의 usekWhAft의 값을 dailyEnergy의 i번째 행의 usekWh 값으로 지정
    # 모든 결측지 값에 대하여 상기 반복

    Anormaly=dailyEnergy[dailyEnergy["Anormaly"]==1][["usekWh"]].reset_index()
    # Anormaly = dailyEnergy에서 Anormaly값이 1인 행의 userkWh열 값만 모은 후 index 재설정
    Anormaly=Anormaly.rename(columns={'datetime' : 'date'})
    # Anormaly = Anormaly에서 datetime 컬럼 명을 date라고 변경

    #anlyUsekWh : 가동중지 추정일 제외한 일별 사용량
    anlyUsekWh=dailyEnergy[["usekWhAft"]].dropna().astype('int64').reset_index()
    # anlyUsekWh = dailyEnergy의 usekWhAft행에서 결측치가 있는 행을 제거한 후 데이터 타입을 정수형으로 바꾼 것의 인덱스를 재설정한 데이터 프레임
    anlyUsekWh=anlyUsekWh.rename(columns={'datetime' : 'date'})
    # anlyUseKwh = anlyUsekWh에서 datetime 컬럼 명을 date라고 변경
    
    dataClust=data_meter["usekWh"]
    # dataClust = data_meter의 usekWh 열

    #timeseries data k-means  -> 시계열 데이터 k-means dataClust를 이용
    ts_value=dataClust.values.reshape(int(len(dataClust)/24),24)
    # ts_value = dataClust의 값 들을 reshape
    # -> dataClust의 길이(데이터 수)를 24개로 나눈 값을 정수로 변환하고, 그 갯수 행, 24열을 가진 2차원 배열 생성 
    # 예를 들어, 총 데이터 포인트가 240개라면, (10일, 24시간) 형태의 배열이 생성됨
    ts_value=np.nan_to_num(ts_value)
    # ts_value는 nan 값을 0으로 반환 

    ############### 왜 nan 값을 0으로 설정하였는지 ###############

    scaler=TimeSeriesScalerMeanVariance(mu=0, std=1)
    # scaler = TimeSeriesScalerMeanVariance(평균이 0이고 표준편차가 1인 scaler)
    data_scaled=scaler.fit_transform(ts_value)
    # data_scaled = ts_value에 scaler를 적용한 것
    data_scaled=np.nan_to_num(data_scaled)
    # data_scaled = data_scaled의 nan 값을 0으로 변환

    ############### 모델 실행 전 결측치 이미 0으로 처리했는데, scaler와 모델 실행 이후 nan값을 왜 0으로 처리하는지? ###############

    #Euclidean k-means 모델 만들기
    km=TimeSeriesKMeans(random_state=0)
    # Km = TimeSeiresKMeans 모델의 약어 (random_state = 0으로 지정)
    visualizer = kelbow_visualizer(km,ts_value, k=(2,9))
    # elbow기법을 통해서 최적의 k 값을 찾기 위함
    # visualizer = kelbow_visualizer(km이라는 모델을 ts_value에 적용하여 k=(2,9) 사이의 값 중 최적의 K 값을 찾는 코드)

    ############### 왜 (2,9)로 범위를 설정하였는지? 근거? ###############

    try:
        number=visualizer.elbow_value_
        km=TimeSeriesKMeans(n_clusters=number, metric="euclidean", verbose=True, random_state=0)
        y_predicted = km.fit_predict(data_scaled)
    except TypeError:
        number = 1
        km=TimeSeriesKMeans(n_clusters=number, metric="euclidean", verbose=True, random_state=0)
        y_predicted = km.fit_predict(data_scaled)
    # number는 visualizer의 elbow 값
    # km은 TimeSeriesKMeans 모델 (클러스터 수는 number개, 방식은 euclidean, verbose는 True, random_state는 0으로 지정)
    # y_predicted은 모델 예측값
    # 만약 TypeError가 발생한다면 number는 1로 지정한 후 
    # 동일하게 모델 돌리기
    # y_predicted는 모델 예측값
    
    # km=TimeSeriesKMeans(random_state=0)
    # visualizer = kelbow_visualizer(km,ts_value, k=(2,9), show=False)
    # number=visualizer.elbow_value_
    # km=TimeSeriesKMeans(n_clusters=number, metric="euclidean", verbose=True, random_state=0)
    # y_predicted = km.fit_predict(data_scaled)
    
    # resultClust : 일별 클러스터링 결과 저장
    resultClust=pd.DataFrame(ts_value.copy())
    # resultClust는 ts_value의 복사본 (데이터 프레임)
    resultClust=pd.concat([resultClust, pd.DataFrame(y_predicted, columns={'predicted'})], axis=1)
    # resultClust는 resultClust에 y_predicted 컬럼을 붙인 데이터 프레임    

    # cluster별 증감 트렌드 도출하기 
    centerClust=pd.DataFrame()
    # centerClust라는 데이터프레임 생성
    countClust=pd.DataFrame()
    # countClust라는 데이터프레임 생성
    for yi in range(number):
        temp=resultClust[resultClust['predicted']==yi].drop(columns='predicted')
        tempMean=pd.DataFrame(round(temp.mean(),1), columns={'{}'.format(yi+1)})
        tempCount=pd.DataFrame({'{}'.format(yi+1) : [len(temp)]})
        
        centerClust=pd.concat([centerClust, tempMean], axis=1)
        countClust=pd.concat([countClust, tempCount], axis=1)
    # 0부터 number-1까지의 수를 yi라고 할때 yi에 대하여 아래 코드 반복 실행
    #   temp=resultClust의 predicted 값이 yi랑 같은 행에 대하여 Predicted 컬럼을 삭제한 후 나머지 값을 저장
    #   tempMean=temp 값의 평균값을 소수점 첫째자리 이하에서 반올림 한 후 데이터 프레임으로 변환하고 열의 이름은 yi+1로 지정
    #   tempCount=temp 데이터프레임의 행 수를 세어 데이터프레임으로 변환하고, 이 데이터프레임의 열 이름을 yi+1로 지정
    #   centerClust=centerClust와 tempMean을 열방향으로 결합
    #   countClust=countClust와 tempCount를 열방향으로 결합


    # 절감 가능액 도출하기 
    # 절감액 : peak 시간 사용량 * 10% 절감 * 전력량요금 단가 * 일수 

    ############### 왜 10% 절감으로 계산하는지?###############

    info_costn=info_costn[(info_costn["item"]==custinfo["ictg"][0])&(info_costn["selCost"]==custinfo["selCost"][0])]
    # info_costn=info_costn에서 item열의 값이 custinfo의 'ictg'의 첫번째 행의 값과 같고, info_costn의 selCost의 값이 custinfo의 selCost의 첫번째 행의 값과 같은 행을 추출하여 새로운 데이터 프레임을 형성

    # avgCost : 전력량요금 평균 단가 
    avgCost=round(info_costn[["kWhCostSF", "kWhCostSummer", "kWhCostWinter"]].mean().mean())
    # avgCost=info_costn의 kWhCostSF, kWhCostSummer, kWhCostWinter 열의 각각 평균을 구한 후 3개 열에 대한 전체 평균값을 구하여 소수점 첫째자리에서 반올림 한 값
    # (전력량 요금의 전체 평균)

    centerDiff=centerClust.diff()
    # centerDiff=CenterClust의 각 열에 대해 현재 행의 값과 이전 행의 값 차이
    centerGuide=pd.DataFrame(index=range(0,24))
    # centerGuide=index가 0에서부터 23까지인 데이터프레임
    for i in centerDiff.columns:
        
        tempCenterDiff=centerDiff[i]
        # incTime : 전력사용량 증가 시간 
        incTime=tempCenterDiff[tempCenterDiff>0]
        # saveTemp : 절감액
        saveTemp=round(incTime*0.1*avgCost*countClust[i][0])
        
        centerGuide=pd.concat([centerGuide, saveTemp], axis=1)

    # centerDiff의 columns를 i라고 할 때 i에 대하여 아래 코드 반복
    #   tempCenterDiff=centerDiff의 i열
    #   incTime=tempCenterDiff의 값이 0보다 큰 행을 추출하여 만든 데이터프레임 -> 전력사용량이 증가하는 시간대를 추출
    #   saveTemp=전력사용량 증가시간*0.1*전력량요금 평균 단가*countClust의 i열의 첫번째 행 값
    #   centerGuide=centerGuide와 saveTemp를 열 기준으로 합친 것
    
    #index가 cluster number가 되도록 transpose
    centerClust=centerClust.transpose()
    # centerClust=centerClust의 행과 열을 바꾼 데이터프레임
    countClust=countClust.transpose().rename(columns={0:'maxTime'})
    # countClust=countclust의 행과 열을 바꾼 데이터프레임에서 열의 값을 0에서 maxTime으로 변경한 데이터프레임
    centerGuide=centerGuide.transpose()
    # centerGuide=centerGuide의 행과 열을 바꾼 데이터프레임
    centerGuide=centerGuide.replace(np.nan, 0)
    # centerGuide=centerGuide의 결측치의 값(nan)을 0으로 바꾼 데이터프레임
    return anlyUsekWh, usekWhStats, Anormaly, centerClust, countClust, centerGuide
    # anlyUsekWh 함수에서 anlyUsekWh, usekWhStats, Anormaly, centerClust, countClust, centerGuide 값을 반환

