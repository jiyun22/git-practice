# -*- coding: utf-8 -*-
"""
Created on Wed May 24 09:08:29 2023

@author: ninewatt
"""

import pandas as pd # 판다스 import
import numpy as np # 넘파이 import
import pyodbc # pyodbc import
from datetime import datetime, timedelta # datetime 패키지에서 datetime, timedelta 모듈 import 
import math # math import

def anlySel(custNo, start, end, info_costn, custInfo, detail_charge):
# anlySel 함수 정의 (인자: custNO, start, end, info_costn, custInfo, detail_charge)
    
    #월별계절정보
    info_month=pd.DataFrame({'month':['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'], 
                             'season':['kWhCostWinter', 'kWhCostWinter', 'kWhCostSF', 'kWhCostSF', 'kWhCostSF', 'kWhCostSummer', 'kWhCostSummer', 'kWhCostSummer', 'kWhCostSF', 'kWhCostSF', 'kWhCostWinter', 'kWhCostWinter']})
    # info_month라는 데이터 프레임 생성
    item=custInfo["ictg"][0]
    # item = custInfo의 ictg열의 첫번째 행 값
    selCost=custInfo['selCost'][0]
    # selCost = custInfo의 selCost열의 첫번째 행 값
    envBill=info_costn[info_costn["item"]=="기후환경요금단가"]["baseCost"].iloc[0]
    # envBill = info_costn의 item열 값이 기후환경요금단가인 행에 대해 baseCost열의 첫번째 행 값 추출
    fuelBill=info_costn[info_costn["item"]=="연료비조정단가"]["baseCost"].iloc[0]
    # fuelBill = info_costn의 item열 값이 연료비조정단가인 행에 대해 baseCost열의 첫번째 행 값 추출
    item_cost=info_costn[info_costn["item"]==item]
    # item_cost = info_costn의 item열 값이 item인 행 추출
    detail_charge["total_usekWh"]=detail_charge[["lload_usekWh","mload_usekWh","maxload_usekWh"]].sum(axis=1)
    # detail_charge의 total_usekWh열 = detail_charge의 lload_usekWh,mload_usekWh,maxload_usekWh열의 합
    for i in range(0, len(detail_charge)):
        detail_charge.at[i, 'date']=detail_charge.iloc[i]['billYm'][0:4]+detail_charge.iloc[i]['billYm'][5:7]
        detail_charge.at[i, 'month']=detail_charge.iloc[i]['billYm'][5:7]
    # 0에서 detail_charge의 데이터 개수까지의 범위를 i라고 할 때 -> i=0, 1, 2, ..., 
    # detail_charge의 i번째 행 date열 = detail_charge의 i+1번째 행의 'billYm'열 값의 0,1,2,3번째 문자열 + detail_charge의 i번째 행의 billYm열 값의 5,6번째 문자열
    # detail_charge의 i번째 행 month열 = detail_charge의 i+1번째 행의 'billYm'열 값의 5,6번째 문자열
    detail_charge.index=detail_charge['date']  
    # detail_charge의 index를 detail_charge의 'date'행으로 설정
    detail_charge=detail_charge.sort_index()
    # detail_charge = detail_charge를 index를 기준으로 데이터 정렬
    detail_charge['avgReqBill']=round(detail_charge["reqBill"].mean(),1)
    # detail_charge의 avgReqBill=detail_charge의 reqBill의 평균 값을 소수점 둘째자리 이하에서 반올림 한 값
    
    #anlySelGraph1 
    anlySelGraph1=detail_charge[["date", "avgReqBill", "lload_usekWh", "mload_usekWh", "maxload_usekWh", "reqBill"]].astype(int)
    # anlySelGraph1=detail_charge의 [["date", "avgReqBill", "lload_usekWh", "mload_usekWh", "maxload_usekWh", "reqBill"]] 열의 형식을 정수형으로 바꾼 것

    #선택요금제 비교
    detail_charge=pd.merge(detail_charge, info_month, how='outer', on='month')
    # detail_charge=detail_charge와 info_month를 'month'열을 기준으로 outer join -> 좌, 우 데이터프레임의 모든 값을 나타냄
    detail_charge=detail_charge.dropna(subset=['reqBill'])
    # detail_charge = detail_charge의 reqBill 열 값이 na 값인 행 제거
    cyber_kepco_cost=pd.DataFrame(columns={"baseBill", "kWhBill", "envBill", "fuelBill"}, index=item_cost["selCost"].unique())
    # cyber_kepco_cost = column이 ("baseBill", "kWhBill", "envBill", "fuelBill")인 데이터프레임 생성 (index는 item_cost의 selCost열의 고유값)
    
    for i in item_cost["selCost"].unique():
    # item_cost의 "selCost"열의 값의 고유값에 대해 i라고 하면 i를 순회하며 아래 코드 반복 실행
        temp=detail_charge[["month", "billAplyPwr", "lload_usekWh", "mload_usekWh", "maxload_usekWh", "season"]].astype(dtype=int, errors='ignore')
        # temp=detail_chatge의 ["month", "billAplyPwr", "lload_usekWh", "mload_usekWh", "maxload_usekWh", "season"] 열의 값 형식을 정수형으로 변경하고 오류 발생시 무시하고 기존의 값으로 반환
        temp=temp.astype({"billAplyPwr":"int", "lload_usekWh":"int", "mload_usekWh":"int", "maxload_usekWh":"int"})
        # temp=temp의 {"billAplyPwr":"int", "lload_usekWh":"int", "mload_usekWh":"int", "maxload_usekWh":"int"}로 형식을 변경
        temp=temp.dropna()
        # temp에서 결측치가 있는 행을 제외하여 temp에 저장
        temp["baseBill"]=temp["billAplyPwr"]*item_cost[item_cost["selCost"]==i]["baseCost"].iloc[0]
        # temp의 baseBill열 = temp의 "billAplyPwr"열 값 * item_cost의 SelCost열 값이 i인 행의 "baseCost"열의 첫번째 행 값
        for j in range(0, len(temp)):
        # 0에서 temp의 데이터의 개수-1까지의 범위의 값을 j라고 할때 j를 순회하며 아래 코드 반복 실행
            temp.at[j, "kWhBill"]=temp.iloc[j]["lload_usekWh"]*item_cost[(item_cost["selCost"]==i)&(item_cost["loadName"]=='1')][temp.iloc[j]["season"]].iloc[0]+temp.iloc[j]["mload_usekWh"]*item_cost[(item_cost["selCost"]==i)&(item_cost["loadName"]=='2')][temp.iloc[j]["season"]].iloc[0]+temp.iloc[j]["maxload_usekWh"]*item_cost[(item_cost["selCost"]==i)&(item_cost["loadName"]=='3')][temp.iloc[j]["season"]].iloc[0]
            # temp의 j번째 행 kWhBill열 값 = temp의 j+1번째 행의 lload_usekWh열 값 * (item_cost의 selCost값이 i이면서, item_cost의 loadName의 값이 1인)
            temp.at[j, "envBill"]=temp.iloc[j][["lload_usekWh", "mload_usekWh", "maxload_usekWh"]].sum()*envBill #기후환경요금단가 : 9원/kWh
            # temp의 j번째 행 envBill열 값 = temp의 j+1번째 행의 ["lload_usekWh", "mload_usekWh", "maxload_usekWh"] 열 값의 합 * envBill
            temp.at[j, "fuelBill"]=temp.iloc[j][["lload_usekWh", "mload_usekWh", "maxload_usekWh"]].sum()*fuelBill #연료비조정단가 : 5원/kWh
            # temp의 j번째 행 fuelBill열 값 = temp의 j+1번째 행의 ["lload_usekWh", "mload_usekWh", "maxload_usekWh"] 열 값의 합 * fuelBill
            temp.at[j, "요금합"]=temp.iloc[j][["baseBill", "kWhBill", "envBill", "fuelBill"]].sum()
            # temp의 j번째 행 요금합 열 값 = temp의 j+1번째 행의 ["baseBill", "kWhBill", "envBill", "fuelBill"]열 값의 합
            temp.at[j, "부가가치세"]=round(temp.iloc[j]["요금합"]*0.1)
            # temp의 j번째 행 부가가치세 열 값 = temp의 j+1번째 행의 "요금합"열 * 0.1
            ########## ###########
            temp.at[j, "전력사업기반기금"]=math.trunc(temp.iloc[j]["요금합"]*0.037/10)*10
            # temp의 j번째 행 전력사업기반자금 열 값 = temp의 j+1번째 행의 "요금합"열 * 0.037/10을 한 값을 내림한 값
        cyber_kepco_cost.at[i, "baseBill"]=temp["baseBill"].sum()
        # cyber_kepco_cost의 i번째 행 baseBill 열 값 = temp의 baseBill 열 값의 합
        cyber_kepco_cost.at[i, "kWhBill"]=temp["kWhBill"].sum()
        # cyber_kepco_cost의 i번째 행 kWhBill 열 값 = temp의 kWhBill 열 값의 합 
        cyber_kepco_cost.at[i, "envBill"]=temp["envBill"].sum()
        # cyber_kepco_cost의 i번째 행 envBill 열 값 = temp의 envBill 열 값의 합
        cyber_kepco_cost.at[i, "fuelBill"]=temp["fuelBill"].sum()
        # cyber_kepco_cost의 i번째 행 fuelBill 열 값 = temp의 fuelBill 열 값의 합
        cyber_kepco_cost.at[i, "요금합"]=temp["요금합"].sum()
        # cyber_kepco_cost의 i번째 행 요금합 열 값 = temp의 요금합 열 값의 합
        cyber_kepco_cost.at[i, "부가가치세"]=temp["부가가치세"].sum()
        # cyber_kepco_cost의 i번째 행 부가가치세 열 값 = temp의 부가가치세 열 값의 합
        cyber_kepco_cost.at[i, "전력사업기반기금"]=temp["전력사업기반기금"].sum()
        # cyber_kepco_cost의 i번째 행 전력사업기반기금 = temp의 전력사업기반기금 열 값의 합
    cyber_kepco_cost=cyber_kepco_cost.astype(float)
    # cyber_kepco_cost = cyber_kepco_cost의 데이터 형식을 실수형으로 변경한 것
    cyber_kepco_cost["totalBill"]=cyber_kepco_cost[["요금합", "부가가치세", "전력사업기반기금"]].sum(axis=1)
    # cyber_kepco_cost의 totalBill 열 = cyber_kepco_cost의 ["요금합", "부가가치세", "전력사업기반기금"]을 각각 열의 합계를 구함
    cyber_kepco_cost["now_cost"]=''
    # cyber_kepco_cost의 now_cost 값을 공백으로 변경
    for i in cyber_kepco_cost.index:
        if i==selCost:
            cyber_kepco_cost.at[i, "now_cost"]=1
        else:
            cyber_kepco_cost.at[i, "now_cost"]=0
    # i는 cyber_kepco_cost의 인덱스를 순회
    # 만약 i가 selCost 값과 같다면
    # cyber_kepco_cost의 i열 now_cost 값에 1 대입
    # 만약 그렇지 않다면
    # cyber_kepco_cost의 i열 now_cost 값에 0 대입

    #tableData
    tableData=cyber_kepco_cost[["baseBill", "kWhBill", "envBill", "fuelBill", "totalBill"]].astype('int64')
    # tableData=cyber_kepco_cost ["baseBill", "kWhBill", "envBill", "fuelBill", "totalBill"] 열의 값들의 형식을 'int64'로 설정
    tableData["selCost"]=cyber_kepco_cost.index
    # tableData의 selCost열 = cyber_kepco_cost의 index로 설정

    pattern_last=detail_charge[detail_charge.index==len(detail_charge)-1]["total_usekWh"] #가장 최근 달의 총 전력사용량
    # pattern_last=detail_charge[detail_charge의 인덱스가 detail_charge의 데이터 갯수-1인 행에 대하여] total_usekWh 열의 값

    pattern_first=detail_charge[detail_charge["date"]==start]["total_usekWh"] #1년전 달의 총 전력사용량
    # pattern_first=detail_charge의 date열의 값이 start인 행의 total_usekWh 열의 값

    if pattern_first.iloc[0]>pattern_last.iloc[0]:
        trend="decrease"
    elif pattern_first.iloc[0]==pattern_last.iloc[0]:
        trend="maintain"
    else:
        trend="increase"
    # 만약 pattern_first의 1번째 행의 값 > pattern_last의 1번째 행의 값이라면
    # trend = "decrease"
    # 만약 pattern_first의 1번째 행의 값 = pattern_last의 1번째 행의 값이라면
    # trend = "maintain"
    # 그렇지 않다면 
    # trend = "increase"
    

    savingCost_opt_cost=min(cyber_kepco_cost[cyber_kepco_cost["now_cost"]==1]["totalBill"])-min(cyber_kepco_cost["totalBill"])
    # savingCost_opt_cost = cyber_kepco_cost의 now_cost행의 값이 1인 행의 totalBill의 최소값-cyber_kepco_cost의 totalBill의 최소값

    if savingCost_opt_cost==0:
        saving='0'
    elif savingCost_opt_cost >=200000:
        saving='up'
    else:
        saving='down'
    # savingCost_opt_cost가 0이라면
    # saving = '0'
    # 만약 savingCost_opt_cost가 200000이상이라면 
    # saving = 'up'
    # 그렇지 않다면 
    # saving = 'down'


    #코멘트1 
    #한번에 json 구조로 구현
    comment1 = {
        "nowPlan":selCost,  #현재요금제
        "optPlan":str(int(cyber_kepco_cost[cyber_kepco_cost["totalBill"]==min(cyber_kepco_cost["totalBill"]
                                                                               )].iloc[0].name)), #최적요금제
        "saveCost":round(savingCost_opt_cost, 4),
        "saving":saving,
        "trend":trend
    }
    # comment1라는 딕셔너리 생성
    ; {"nowPlan":selCost,  #현재요금제
    # nowPlan : selCost
    ; "optPlan":str(int(cyber_kepco_cost[cyber_kepco_cost["totalBill"]==min(cyber_kepco_cost["totalBill"])].iloc[0].name)), #최적요금제
    # optPlan : cyber_kepco_cost의 totalBill열의 값이 cyber_kepco_cost의 totalBill열의 최소값과 같은 행들 중 1번째 행의 이름을 정수형으로 변환 후 문자형으로 변환
    ; "saveCost":round(savingCost_opt_cost, 4),
    # saveCost : savingCost_opt_cost를 소수점 다섯째 자리 이하에서 반올림 한 값
    ; "saving":saving,
    # saving : saving
    ; "trend":trend}
    # trend : trend
    

    #anlySelGraph2
    detail_charge["useTime"]=round(detail_charge["total_usekWh"]/detail_charge["billAplyPwr"],1)  #월평균사용시간
    # detail_charge의 useTime열 = (detail_charge의 total_usekWh열 값 / detail_charge의 billAplyPwr의 값)을 소수점 둘째자리 이하에서 반올림한 값
    detail_charge["avgUseTime"]=round(detail_charge["useTime"].mean(),1)    #연평균사용시간
    # detail_charge의 avgUseTime열 = detail_charge의 useTime열의 평균값을 소수점 둘째자리 이하에서 반올림한 값
    anlySelGraph2=detail_charge[["avgUseTime", "date", "useTime"]]
    # anlySelGraph2 = detail_charge의 ["avgUseTime", "date", "useTime"]열

    #코멘트2
    comment2 = {
            "avgUseTime" : round(detail_charge["avgUseTime"][0], 1), 
            "maxTime" : round(detail_charge["useTime"].max(), 1), 
            "maxMonth" : detail_charge[detail_charge["useTime"]==detail_charge["useTime"].max()]["date"].iloc[0][4:7], 
            "minTime" : round(detail_charge["useTime"].min(), 1), 
            "minMonth" : detail_charge[detail_charge["useTime"]==detail_charge["useTime"].min()]["date"].iloc[0][4:7]
    }
    # comment2라는 딕셔너리 생성
           {    
            "avgUseTime" : round(detail_charge["avgUseTime"][0], 1), 
            "maxTime" : round(detail_charge["useTime"].max(), 1), 
            "maxMonth" : detail_charge[detail_charge["useTime"]==detail_charge["useTime"].max()]["date"].iloc[0][4:7], 
            "minTime" : round(detail_charge["useTime"].min(), 1), 
            "minMonth" : detail_charge[detail_charge["useTime"]==detail_charge["useTime"].min()]["date"].iloc[0][4:7]
    }     
    selFare=item_cost[item_cost["selCost"]==selCost]
    selFare=selFare.reset_index(drop=True)
    
    return anlySelGraph1, tableData, comment1, anlySelGraph2, comment2, selFare


