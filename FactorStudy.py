# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 13:15:06 2020

@author: Jo

Classical factor study methods
"""

import math
import pandas as pd
import numpy as np
from datetime import datetime
from calendar import monthrange
from datetime import date as Date
import matplotlib.pyplot as plt
from tqdm import tqdm
import statsmodels.api as sm

def load_data(directory, file_name, file_format='csv'):
    """
    载入数据
    :param directory: string 文件所在目录, 如'F:\SJTU_文件'
    :param file_name: string 文件名, 如'CHN2000'
    :param file_format: string 文件后缀（'csv','xlsx','xls','txt'）, 默认为'csv'
    :return: dataframe
    """
    if file_format == 'csv':
        return pd.read_csv("%s\\%s.%s" % (directory, file_name, file_format))
    elif file_format in ('xlsx', 'xls'):
        return pd.read_excel("%s\\%s.%s" % (directory, file_name, file_format))
    elif file_format == 'txt':
        return pd.read_table("%s\\%s.%s" % (directory, file_name, file_format))
    else:
        print('请输入 %s 文件正确的格式' % (file_name))
    
def extract_data(data, index, columns, values, is_month=0):
    """
    提取多股票时间序列数据
    :param data: dataframe 包含多期多股票
    :param index: string 变量名（日期）
    :param columns: string 变量名（股票代码）
    :param values: string 变量名
    :param is_month: bool 是否为月度数据, 默认为0（日度数据）
    :return: dataframe index为日期（格式'2000-01-01'），columns为股票代码
    """
    data = data.pivot(index=index, columns=columns, values=values)
    if is_month:
        def f(date):
            try:
                date = datetime.strptime(date, '%Y-%m-%d')
                monthend = monthrange(date.year, date.month)[1]
                return (Date(date.year, date.month, monthend)).strftime('%Y-%m-%d')
            except:
                date = datetime.strptime(date, '%Y-%m')
                monthend = monthrange(date.year, date.month)[1]
                return (Date(date.year, date.month, monthend)).strftime('%Y-%m-%d')
            else:
                print("请保证月度日期格式为'2000-01-01'或'2000-01'")
        data.index = map(lambda x:f(x), data.index)
    data.columns = map(lambda x:str(x), data.columns)
    return data

def univar_sort(factorvar, group_num):
    """
    单变量排序构造股票组
    :param factorvar: dataframe index为日期, columns为股票代码，value为因子变量
    :param group_num: int 股票组数
    :return: dict key为日期，value为datadrame index为序号，columns为组号 1-组数，value为股票代码
    """
    Groups = dict()
    for i in tqdm(range(len(factorvar))):
        try:
            var = factorvar.iloc[i,:].dropna()
            var_q = np.zeros(group_num+1)
            var_q[0] = var.min()-1
            groups = pd.Series(dtype='float64')
            for j in range(group_num):
                var_q[j+1] = np.percentile(var, (j+1)*(100/group_num))
                D = var[(var>var_q[j])&(var<=var_q[j+1])].index.values.tolist()
                groups = pd.concat([groups, pd.Series(D)],axis=1) if not groups.empty else pd.Series(D)
            groups.columns = range(1, group_num+1)
            Groups.update({factorvar.index[i]:groups})
        except:
            print("错误：%s" % (factorvar.index[i]))
    return Groups

def double_sort_i(var1, var2, num1, num2):
    """
    独立双重分组
    :param var1: dataframe index为日期, columns为股票代码，value为因子变量
    :param var2: dataframe index为日期, columns为股票代码，value为因子变量 
    :param num1: int 变量1组数
    :param num2: int 变量2组数
    :return: dict key为日期，value为datadrame index为序号，columns为组号 1-组数，value为股票代码
    股票组排序依次为var1_min&(var2_min~var2_max) ~ var1_max&(var2_min~var2_max)
    """
    Groups = dict()
    datelist = list(set(var1.index)&set(var2.index))
    datelist.sort()
    
    for i in tqdm(range(len(datelist))):
        try:
            var_1 = var1.iloc[i,:].dropna()
            var1_q = np.zeros(num1+1)
            var1_q[0] = var_1.min()-1
            var1_groups = pd.Series(dtype='float64')
            
            var_2 = var2.iloc[i,:].dropna()
            var2_q = np.zeros(num2+1)
            var2_q[0] = var_2.min()-1
            var2_groups = pd.Series(dtype='float64')
            
            for j in range(num1):
                var1_q[j+1] = np.percentile(var_1, (j+1)*(100/num1))
                D = var_1[(var_1>var1_q[j])&(var_1<=var1_q[j+1])].index
                var1_groups = pd.concat([var1_groups, pd.Series(D)],axis=1) if not var1_groups.empty else pd.Series(D)
            var1_groups.columns = range(1, num1+1)
            
            for j in range(num2):
                var2_q[j+1] = np.percentile(var_2, (j+1)*(100/num2))
                D = var_2[(var_2>var2_q[j])&(var_2<=var2_q[j+1])].index
                var2_groups = pd.concat([var2_groups, pd.Series(D)],axis=1) if not var2_groups.empty else pd.Series(D)
            var2_groups.columns = range(1, num2+1)
            
            groups = pd.Series(dtype='float64')
            for m in range(1, num1+1):
                for n in range(1, num2+1):
                    g1 = var1_groups[m].dropna()
                    g2 = var2_groups[n].dropna()
                    G = pd.Series(list(set(g1)&set(g2)))
                    groups = pd.concat([groups, G], axis=1) if not groups.empty else G
            groups.columns = range(1, num1*num2+1)
            Groups.update({datelist[i]:groups})
        except:
            pass
    return Groups

def double_sort_d(var1, var2, num1, num2):
    """
    依赖双重分组
    :param var1: dataframe index为日期, columns为股票代码，value为因子变量
    :param var2: dataframe index为日期, columns为股票代码，value为因子变量 
    :param num1: int 变量1组数
    :param num2: int 变量2组数
    :return: dict key为日期，value为datadrame index为序号，columns为组号 1-组数，value为股票代码
    股票组排序依次为var1_min&(var2_min~var2_max) ~ var1_max&(var2_min~var2_max)
    """
    Groups = dict()
    datelist = list(set(var1.index)&set(var2.index))
    datelist.sort()
    
    for i in tqdm(range(len(datelist))):
        try:
            var_1 = var1.iloc[i,:].dropna()
            var1_q = np.zeros(num1+1)
            var1_q[0] = var_1.min()-1
            var1_groups = pd.Series(dtype='float64')
            groups = pd.Series(dtype='float64')
            for j in range(num1):
                var1_q[j+1] = np.percentile(var_1, (j+1)*(100/num1))
                D = var_1[(var_1>var1_q[j])&(var_1<=var1_q[j+1])].index
                
                var_2 = var2.loc[datelist[i], list(set(D)&set(var2.columns))].dropna()
                var2_q = np.zeros(num2+1)
                var2_q[0] = var_2.min()-1
                for k in range(num2):
                    var2_q[k+1] = np.percentile(var_2, (k+1)*(100/num2))
                    E = var_2[(var_2>var2_q[k])&(var_2<=var2_q[k+1])].index
                    groups = pd.concat([groups, pd.Series(E)], axis=1) if not groups.empty else pd.Series(E)
            groups.columns = range(1, num1*num2+1)
            Groups.update({datelist[i]:groups})
        except:
            pass
    return Groups

def cal_raw_return(groups, holding, returns, weightvalues=pd.DataFrame()):
    """
    计算股票组合（可持股多期）未经风险调整的月度收益率
    :param groups: dict key为日期，value为datadrame index为序号，columns为组号 1-组数，value为股票代码
    :param holding: int 股票组合持有月数 
    :param returns: dataframe index为日期，columns为股票代码，value为收益率
    :param weightvalues: dataframe index为日期，columns为股票代码，value为加权变量
    :return: dateframe index为日期，columns为组号，value为收益率
    
    【注】参考Jegadeesh&Titman(1993)，每期同时持有多个股票组合，各股票组合可以采用总市值加权
    或者其他加权方式，股票组合之间采用等权重。
    """
    datelist = list(groups.keys())
    group_num = groups[datelist[0]].shape[1]
    datelist = list(set(datelist) & set(returns.index))
    datelist.sort()
    returns = returns.loc[datelist, :]

    group_rawret = dict()    
    if weightvalues.empty:    
        for k in tqdm(range(1, group_num+1)):
            rawret = pd.Series(dtype='float64')
            for i in range(1, len(returns)):
                try:
                    D = returns.loc[datelist[i:i+holding], list(groups[datelist[i-1]][k].dropna())] \
                    .apply(lambda x:x.mean(), axis=1)
                    rawret = pd.concat([rawret, D], axis=1) if not rawret.empty else D    
                except:
                    pass
            group_rawret.update({k:rawret})
    else:
        for k in tqdm(range(1, group_num+1)):
            rawret = pd.Series(dtype='float')
            for i in range(1, len(returns)):
                try:
                    D = returns.loc[datelist[i:i+holding], list(groups[datelist[i-1]][k].dropna())] 
                    E = weightvalues.loc[datelist[i-1], list(groups[datelist[i-1]][k].dropna())] 
                    weight = E/E.sum()
                    F = D.apply(lambda x:x*weight, axis=1).apply(lambda x:x.sum(), axis=1)
                    rawret = pd.concat([rawret, F], axis=1) if not rawret.empty else F
                except:
                    pass
            group_rawret.update({k:rawret})
    
    Group_rawret = pd.Series(dtype='float64')
    for i in range(1, group_num+1):
        D = group_rawret[i].apply(lambda x:x.mean(), axis=1)
        Group_rawret = pd.concat([Group_rawret, D],axis=1) if not Group_rawret.empty else D
    Group_rawret.columns = range(1, group_num+1)
    
    return Group_rawret

def cal_alphabeta(assets, factormodel=pd.DataFrame()):
    """
    计算股票组合的alpha和因子暴露β(时序回归)    
    :param assets: dataframe index为日期，columns为各股票组名称 values为股票组收益率
    :param factormodel: dataframe index为日期，columns依次为RF和因子名称 value为RF和因子收益率
    :return: dataframes index为alpha,t-stat，columns为各股票组名称；index为beta，columns为股票组，
    values为beta值；index为t，columns为股票组，values为t值
    
    【注】：t-stat经过Newey&West(1987)调整。
    """
    def f(x):
        data = pd.concat([x, factormodel], axis=1).dropna()
        data = sm.add_constant(data, False)
        T = len(data)
        L = math.floor(4*(T/100)**(2/9))
        results = sm.OLS(data.iloc[:,0]-data.iloc[:,1], data.iloc[:,2:]).fit(cov_type='HAC', cov_kwds={
                'maxlags':L})
        return results.params[-1], results.tvalues[-1], results.params[:-1], results.tvalues[:-1]
    
    def g(x):
        data = sm.add_constant(x.dropna(), False)
        T = len(data)
        L = math.floor(4*(T/100)**(2/9))
        results = sm.OLS(data.iloc[:,0], data.iloc[:,1]).fit(cov_type='HAC', cov_kwds={'maxlags':L})
        return results.params[-1], results.tvalues[-1]
    
    assets = pd.DataFrame(assets)
    coef = assets.apply(lambda x:f(x)) if not factormodel.empty else assets.apply(lambda x:g(x))
    alpha = coef.iloc[:2,:]
    alpha.index = ['alpha', 't-stat']
    if not factormodel.empty:
        beta = pd.Series(dtype='float64')
        for D in coef.iloc[2,:]:
            beta = pd.concat([beta, D], axis=1) if not beta.empty else D
        beta.index = list(map(lambda x:'β-'+x, beta.index))
        beta.columns = coef.columns
        
        beta_t = pd.Series(dtype='float64')
        for E in coef.iloc[3,:]:
            beta_t = pd.concat([beta_t, E], axis=1) if not beta_t.empty else E
        beta_t.index = list(map(lambda x:'t-'+x, beta_t.index))
        beta_t.columns = coef.columns
        return alpha, beta, beta_t
    return alpha

def cross_regression(assets, loadings, choice, wls=False, weights=None):
    """
    截面回归  
    :param assets: dataframe index为日期，columns为各股票(组)名称 values为股票(组)收益率
    :param loadings: dict key为因子名称，value为dataframe index为日期，columns为各股票（组）
    名称，values为因子暴露值
    :param choice: list 作为自变量的因子名称
    :param wls: 是否使用加权最小二乘回归WLS
    :param weights: dataframe index为日期，columns为各股票(组)名称 values为WLS的权重变量
    :return: dataframes index为截面回归系数时序平均值,t-stat，columns为因子名称
    
    【注】：t-stat经过Newey&West(1987)调整。
    """

    assets = assets.shift(-1).dropna(how='all') # t+1期的收益率和t期的变量截面回归
    datelist = list(assets.index)
    var = list(loadings.keys())
    for i in var:
        datelist = list(set(datelist)&set(loadings[i].dropna(how='all').index))
        datelist.sort()
    assets = assets.loc[datelist,:]
    
    def f(x):
        try:
            data = x
            for i in var:
                data = pd.concat([data, loadings[i].loc[x.name,:]], axis=1)
            data.columns = ['R']+var
            data = sm.add_constant(data.dropna(), False)
            if wls:
                weight = weights.loc[x.name,data.index]
                return sm.WLS(data.iloc[:,0], data.loc[:,choice+['const']], weights=weight).fit().params
            else:
                return sm.OLS(data.iloc[:,0], data.loc[:,choice+['const']]).fit().params
        except:
            pass

    coef = assets.apply(lambda x:f(x), axis=1)
    
    def g(x):
            data = sm.add_constant(x.dropna(), False)
            T = len(data)
            L = math.floor(4*(T/100)**(2/9))
            results = sm.OLS(data.iloc[:,0], data.iloc[:,1]).fit(cov_type='HAC', cov_kwds={'maxlags':L})
            return results.params[-1], results.tvalues[-1]
    factorret = coef.apply(lambda x:g(x))
    factorret.index=['coef','t']
    
    return factorret




if __name__ == '__main__':
    
    #read data
    
    SIZE = pd.read_csv('.\data\SIZE.csv').set_index('Unnamed: 0')
    PE = pd.read_csv('.\data\PE.csv').set_index('Unnamed: 0')
    BETA = pd.read_csv('.\data\BETA.csv').set_index('Unnamed: 0')
    MAX = pd.read_csv('.\data\MAX.csv').set_index('Unnamed: 0')
    ff3 = pd.read_csv('.\data\FF3_monthly.csv').set_index('date')
    Mret = pd.read_csv('.\data\Mret.csv').set_index('Unnamed: 0')
    Msize = pd.read_csv('.\data\Msize.csv').set_index('Unnamed: 0')
    

    # group stocks by MAX to 5 groups
    
    g5 = univar_sort(MAX,5)
    
    # group stocks by SIZE and EP to 2*3 groups 

    PE[PE<0] = np.nan # delete the negative PE
    EP = 1/PE
    
    # independent
    g_i = double_sort_i(SIZE, EP, 2, 3)        
    
    # dependent
    g_d = double_sort_d(SIZE, EP, 2, 3)
    
    # calculate the raw returns of each group 
    g5_ret_ew = cal_raw_return(g5, 1, Mret) # equal-weighted holding 2 months 
    g5_ret_vw = cal_raw_return(g5, 1, Mret, Msize) # value-weigted    
        
    # construct the size and value factor applying the FF's way
    # SMB = (SL+SM+SH)/3 - (BL+BM+BH)/3
    # HML = (HS+HB)/2 - (LS+LB)/2
    gi_ret_vw = cal_raw_return(g_i, 1, Mret, Msize)
    size = gi_ret_vw.iloc[:,:3].mean(axis=1) - gi_ret_vw.iloc[:,3:].mean(axis=1)
    hml = (gi_ret_vw[3]+gi_ret_vw[6])/2 - (gi_ret_vw[1]+gi_ret_vw[4])/2
    
    # calculate alpha relative to factor models
    
    ff3['MKT'] = ff3['MKT']-ff3['RF']
    FM = ff3.copy()
    LSret = g5_ret_ew[5]-g5_ret_ew[1]
    alpha = cal_alphabeta(LSret)
    
    # calculate beta relative to factor models
    alpha,beta,beta_t = cal_alphabeta(LSret,FM)
    
    # Fama-MacBeth regression
    regressors = {'MAX':MAX, 'BETA':BETA, 'EP':EP, 'SIZE':SIZE}
    choice = list(regressors.keys())
    res = cross_regression(Mret, regressors, choice)
    res_wls = cross_regression(Mret, regressors, choice, True, np.sqrt(Msize))
    
    
    
ffc = pd.read_csv('.\data\FFC_monthly.csv').set_index('date') 
IV = pd.read_csv('.\data\Factor\Characteristics\IV.csv').set_index('Unnamed: 0')
g5 = univar_sort(IV,5)
raw = cal_raw_return(g5, 1, Mret)
raw.mean()
alpha = cal_alphabeta(raw,FFC)










   
    
    
    
    
    
    
    
    
    
    
    
    
    
