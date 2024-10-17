# -*- coding: utf-8 -*-
"""
Created on Mon May 22 10:41:48 2023

@author: ninewatt
"""
import pandas as pd # 판다스 import
import numpy as np # 넘파이 import
from utils.res_utils import res_fail, res_ok # utils.res_utils라는 패키지에서 res_fail, res_ok 모듈 import 
 
def anlyPV(data, data2):  # anlyPV 함수 정의 (인자: data, data2)  
    data.drop(data[(data["grndFlrCnt"] == 0)].index, inplace=True) # data에서 grndFlrCnt 열의 값이 0인 행의 index를 제거하여 data에 저장
    data.drop(data[(data["vlRatEstmTotArea"] == 0)].index, inplace=True) # data에서 vlRatEstmTotArea 열의 값이 0인 행의 index를 제거하여 data에 저장
    data["지붕면적"]=data["vlRatEstmTotArea"]/data["grndFlrCnt"] # data에 "지붕면적" 열 생성 -> data의 vlRatEstmTotArea 행 값 / data grndFlrCnt 행 값
    data["태양광용량"]=round(data["지붕면적"]/9.9, 2) # data에 "태양광용량" 열 생성 -> data의 지붕면적 값 / 9.9 한 값을 소수점 셋째자리 이하 반올림
    #3평(9.9m2) 당 1kW

    ############### 산출식 근거? ###############

    data["예상발전량"]=round(data["태양광용량"]*3.6*365) # data에 "예상발전량" 열 생성 -> data의 태양광용량 열 값 * 3.6 * 365를 소수점 이하 반올림 한 값
    #설비용량(kW)*3.6hr(하루 평균 발전시간)*365일

    ############### 산출식 근거? ###############

    data["공사비"]=round(data["태양광용량"]*1300000).astype(int) # data에 "공사비" 열 생성 -> data의 태양광용량 열 값 * 1300000를 소수점 이하 반올림하고 데이터 형식을 정수형으로 설정
    #설비용량(kW)*130만원(최소기준)

    ############### 산출식 근거? ###############

    data["판매형_수익"]=round(data["예상발전량"]*160).astype(int) # data에 "판매형_수익" 열 생성 -> data의 예상발전량 열 값 * 160울 소수점 이하 반올림하고 데이터 형식을 정수형으로 설정   
    #예상발전량(kWh/yr)*전력판매단가(160원/kW)

    ############### 산출식 근거? ###############

    data["판매형_ROI"]=round(data["공사비"]/data["판매형_수익"],1) # data에 "판매형_ROI" 열 생성 -> data의 공사비 / data의 판매형_수익을 소수점 둘째자리 이하 반올림
    
    eul=pd.DataFrame({"itcg" : ["226","236","430","431","432","526","536","610","726","736","746","910","915"]})
    # eul="itcg"열 값이 각각 ["226","236","430","431","432","526","536","610","726","736","746","910","915"]인 데이터프레임 생성

    if len(eul[eul["itcg"]==data2.ictg[0]])>0:  #계약종별이 을인 경우, 태양광용량(kW)*3.6hr(하루평균 발전시간)*연간 건물 운영일(주5일 기준)* 발전 시간대 한전 전력 절감 단가 
        data["자가소비형_수익"]=round(data["태양광용량"]*3.6*264*165).astype(int)  #계약전력(을) 기준 고압A 선택II 기준
    else:
        data["자가소비형_수익"]=round(data["태양광용량"]*3.6*264*129).astype(int)  #계약전력(갑) 기준 고압A 선택II 기준 
    data["자가소비형_ROI"]=round(data["공사비"]/data["자가소비형_수익"],1)
    
    # 만약 eul의 itcg값이 data2의 ictg값의 첫번째 행 값과 같은 데이터의 개수가 1개 이상이면 
    # data의 "자가소비형_수익"열 값 -> data 태양광용량 값*3.6*264*165를 소수점 이하 반올림하고 데이터 형식을 정수형으로 설정한 값
    # 그렇지 않으면
    # data의 "자가소비형_수익"열 값 -> data 태양광용량 값*3.6*264*129를 소수점 이하 반올림하고 데이터 형식을 정수형으로 설정한 값
    # data의 "자가소비형_ROI"열 값 -> data 공사비 값 / data 자가소비형_수익 값을 소수점 둘째자리 이하에서 반올림한 값

    PVtable1 = {
            "PVvol" : data["태양광용량"].sum(),
            "preGen" : data["예상발전량"].sum(),
            "cost" : data["공사비"].sum()
            }
    # PVtable1 = {"PVvol" : 태양광용량의 합, "preGen" : 예상발전량의 합, "cost" : 공사비의 합} 인 딕셔너리 생성

    PVtable2 = {
                "selfCost" : data["자가소비형_수익"].sum(),
                "selfROI" : data["자가소비형_ROI"].iloc[0],
                "sellCost" : data["판매형_수익"].sum(),
                "sellROI" : data["판매형_ROI"].iloc[0]
                }
     # PVtable2 = {"selfCost" : 자가소비형_수익의 합, "selfROI" : 자가소비형_ROI의 첫번째 행 값, "sellCost" : 판매형_수익의 합, "sellROI" : 판매형_ROI의 첫번째 행 값} 인 딕셔너리 생성
    
    PVanlysis = {
        "area" : round(data["지붕면적"].sum(),2),
        "capacity" : round(round(data["지붕면적"]*0.15, 2).sum(),2)
    }
    # PVanlysis = {"area" : 지붕면적의 합을 소수점 셋째자리 이하에서 반올림 한 값, "capacity" : 지붕면적*0.15를 소수점 셋째자리 이하에서 반올림한 값의 합을 소수점 셋째자리 이하에서 반올림한 값} 인 딕셔너리 생성

    return PVtable1, PVtable2, PVanlysis
    # PVtable1, PVtale2, PVanlysis 반환