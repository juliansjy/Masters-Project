# from centralConstants import *
import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt;

m = 1.2043
a = 49.1298
b = 0.2012
demand = 1468000; # raw data demand in 2050 from source. required to calc demandCAL
ms1 = 0.4
ms2 = 0.18
thou = 1000
daysyear = 365
cons1 = 8760
cons2 = 1.07
cons3 = 16.92
cons4 = 22.43
cons5 = 15
mill = 1000000

plantOperationalCap = 0.95;
CO2emissionRate = 1.06; # Kg of CO2 per Kg of H2
CO2captureRate = 8.60; # Kg of CO2 per Kg of H2 
CO2emissionRatefeedstock =  0.28;
CO2emissionRatefuel = 1.18;
NatGasConsumption = 33578;
plantDesignCap = 190950;

# dailyprodrate = 190000;
# yearlyprodrate = dailyprodrate * plantOperationalCap * (365/1000);
Hprice = 11.5;

iniProdRateDay = 47500
iniProdRateYear = iniProdRateDay*plantOperationalCap*365/1000



discountrate = 0.1; # Nine percent per annum
time = 25;
statetax = 0.0725;
fedtax = 0.21;

basecostFF = 67836966;
basecostWM = 104434;
basecostCC = 625712;

#Constants for Flexible and Phased models
#Full scale capacity
dailycapreq = 190000

expansionIncrement = (iniProdRateDay*plantOperationalCap*365)/1000 #replace in code 16470.63
expansionTimes = ((dailycapreq*plantOperationalCap*365/1000)-(iniProdRateDay*plantOperationalCap*365/1000))/((iniProdRateDay*plantOperationalCap*365)/1000) #replace in code 3
expansionThresh = 0.75

#expansion options
reductionDuringExpansion = 0.5

#production factors
eos = 0.68
learningRate = 0
initialCapCostsFixed = 591.73
initialCapCostsPhased = 266.34

dict = {
    16470.63: 6.59,
    32941.26: 10.56,
    49411.89: 13.91,
    65882.52: 16.92,
}

expansionInvestmentval = 137.12

# dict2 = {
#     16470.63: 137.12,
#     32941.26: 137.12,
#     49411.89: 137.12,
#     65882.52: 137.12,
# }

th = 25

MACRSfixed = [22,43,40,37,34,31,29,27,26,26,26,26,26,26,26,26,26,26,26,26,13,0,0,0,0]
MACRSphased = [10,19,18,16,15,19,23,21,20,25,29,28,27,31,35,34,33,32,31,31,24,18,18,18,18]
MACRSflexible = [10,19,18,16,15,14,13,12,12,17,22,21,20,25,29,33,36,35,34,33,26,19,18,18,18]
q = [46.96, 50, 54.25, 58.86, 63.86, 69.29, 75.18, 81.57, 88.51, 96.03, 104.19, 113.05, 122.66, 133.08, 144.40, 156.67, 169.99, 184.44, 200.11, 217.12, 235.58, 255.60, 277.33, 300.90, 326.48]
nomyear = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
nums = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
year = [2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035, 2036, 2037, 2038, 2039, 2040, 2041, 2042, 2043, 2044, 2045, 2046, 2047, 2048, 2049, 2050]
expansionChange = [0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]
# df = pd.read_excel('../../data.csv');

# start of calcs

#Finding Demand per year
def test(help):
    iniProdRateDay = help[0]
    # expansionIncrement = help[1]
    reductionDuringExpansion = help[1]
    
# year = df.iloc[0:27, 0]; # years imported from csv file from 2023-2049
    # demand = 1468000; # raw data demand in 2050 from source. required to calc demandCAL
    demandCAL = pd.Series(index=nums, dtype='float64')
    # newList = pd.Series(index=nums)
    availableRate = pd.Series(index=nums, dtype='float64')
    finalprodrate = pd.Series(index=nums, dtype='float64')
    ProdCO2emissions = pd.Series(index=nums, dtype='float64')
    CO2capture = pd.Series(index=nums, dtype='float64')
    upstreamCO2emissions = pd.Series(index=nums, dtype='float64')
    revenue = pd.Series(index=nums, dtype='float64')
    fixedOPEX = pd.Series(index=nums, dtype='float64')
    varOPEXcost = pd.Series(index=nums, dtype='float64')
    OpCost = pd.Series(index=nums, dtype='float64')
    Depreciation = pd.Series(index=nums, dtype='float64')
    expansionInvestment = pd.Series(index=nums, dtype='float64')
    # expansionInvestment2 = pd.Series(index=nums, dtype='float64')
    sv = 0
    # sv2 = 0
    cf = pd.Series(index=nums, dtype='float64')
    dcf = pd.Series(index=nums, dtype='float64')
    npv1 = 0
    npv = 0

    for i in range(len(year)):
        demandCAL[i] = (m/(1+a*np.exp(-b*(year[i]-year[0])))*demand)*ms1*ms2 #change p to m and in the centralConstants file
    # demandCAL = newList.array
    # BlueH2prodCAL = demandCAL*ms1*ms2;
    demandSF = demandCAL[2:27]; #start from index 2 to 26 instead of 0 to 24

    # newList = (m/(1+a*np.exp(-b*(year-year[0])))*demand).drop([0,1]) #list of demand amounts per year 
    # demandCAL = newList.array #convert demand list to an array

    # BlueH2prodCAL = demandCAL*0.4;

    # demandSF = BlueH2prodCAL*0.18 #final demand used in all calculations

    #Available Rate

    # availableRate = []
    # expansionChange = df.iloc[2:27, 8].array

    for i in range(0, th):
        availableRate[0] = iniProdRateYear
        if expansionChange[i] == 1 and availableRate[i] < (iniProdRateYear+((((dailycapreq*plantOperationalCap*365/1000)-(iniProdRateDay*plantOperationalCap*365/1000))/((iniProdRateDay*plantOperationalCap*365)/1000))*((iniProdRateDay*plantOperationalCap*365)/1000))):
            availableRate[i+1] = ((iniProdRateDay*plantOperationalCap*365)/1000) + availableRate[i]
        elif expansionChange[i] == 0:
            availableRate[i+1] = availableRate[i]
    availableRate2 = availableRate[0:25]

    # availableRate.append(iniProdRateYear)

    # for i, v in enumerate(demandSF):
    #     if expansionChange[i] == 1 and availableRate[i] < (iniProdRateYear+(expansionTimes*expansionIncrement)):
    #         availableRate.append(expansionIncrement + availableRate[i])
    #     elif expansionChange[i] == 0:
    #         availableRate.append(availableRate[i])

    # availableRate.pop()
            

    #Final Production Rate

    # finalprodrate = []

    for i in range(0,th):
        if expansionChange[i-1] == 1:
            finalprodrate[i] = min(demandSF[i+2], availableRate2[i])*(1-reductionDuringExpansion)
        elif expansionChange[i-1] == 0:
            finalprodrate[i] = min(demandSF[i+2], availableRate2[i])

    # for i, v in enumerate(demandSF):
    #     if expansionChange[i-1] == 1:
    #         finalprodrate.append(min(demandSF[i], availableRate[i])*(1-reductionDuringExpansion))
    #     elif expansionChange[i-1] == 0:
    #         finalprodrate.append(min(demandSF[i], availableRate[i]))

    #Production of CO2 emission, CO2 capture, Upstream and Distribution of CO2 emissions

    # ProdCO2emissions = [];
    # CO2capture = []
    # upstreamCO2emissions = [];

    for i in range(0, th):
        ProdCO2emissions[i] = CO2emissionRate*(finalprodrate[i]*thou)/thou
        CO2capture[i] = CO2captureRate*(finalprodrate[i]*thou)/thou
        upstreamCO2emissions[i] = (CO2emissionRatefeedstock/thou)*(NatGasConsumption*cons1)*(finalprodrate[i]/(plantDesignCap*plantOperationalCap*daysyear/thou))

    # for j in finalprodrate:
    #     ems = CO2emissionRate*(j*1000)/1000
    #     capture = CO2captureRate*(j*1000)/1000
    #     upstream = (CO2emissionRatefeedstock/1000)*(NatGasConsumption*8760)*(j/(plantDesignCap*plantOperationalCap*365/1000))
    #     ProdCO2emissions.append(ems)
    #     CO2capture.append(capture)
    #     upstreamCO2emissions.append(upstream)

    #Revenues
    for i in range(0, th):
        revenue[i] = Hprice*(finalprodrate[i]*thou)/mill

    # revenue = [];

    # for j in finalprodrate:
    #     revenue.append(Hprice*(j*1000)/1000000);
        
    # rev = pd.Series(revenue).array

    #Capital Investment
    #capInvestment = 591730751/1000000 need to change this


    # expansionInvestment = df.iloc[2:27, 9].array;
    # capitalInvesment = expansionInvestment;

    for i in range(0,th):
        if availableRate2[i] >= 0 and availableRate2[i] <= 16470.63:
            fixedOPEX[i] = dict[16470.63]
        elif availableRate2[i] > 16470.63 and availableRate2[i] <= 32941.26:
            fixedOPEX[i] = dict[32941.26]
        elif availableRate2[i] > 32941.26 and availableRate2[i] <= 49411.89:
            fixedOPEX[i] = dict[49411.89]
        elif availableRate2[i] > 49411.89 and availableRate2[i] <= 65882.52:
            fixedOPEX[i] = dict[65882.52]
        elif availableRate2[i] > 65882.52:
            fixedOPEX[i] = dict[65882.52]

    # for i in range(0, th):
    #     for key, val in dict.items():
    #         if 0 <= key in availableRate2[i] <= 2*availableRate2[i]:
    #             fixedOPEX[i] = val
        # fixedOPEX[i] = dict[availableRate2[i]]

    # for key, val in dict.items():
    #     if key <= availableRate2[i] or key >= availableRate2[i]:
    #         fixedOPEX[i] = val

    #Operational Costs

    NG = pd.Series(index=nums, dtype='float64')
    WM = pd.Series(index=nums, dtype='float64')
    CC = pd.Series(index=nums, dtype='float64')
    CO2tax = pd.Series(index=nums, dtype='float64')
    CO2TS = pd.Series(index=nums, dtype='float64')
    H2delivery = pd.Series(index=nums, dtype='float64')
    CO2tax = pd.Series(index=nums, dtype='float64')
    CO2TS = pd.Series(index=nums, dtype='float64')

    # NG = []; #Natural Gas
    # WM = []; #Water Makeup
    # CC = []; #Chemicals and Catalysts
    # CO2tax = []; #CO2 tax
    # CO2TS = []; #CO2 Transport and Storage Costs
    # H2delivery = []; #H2 Delivery Costs

    # fixedOPEX = []; #Fixed OPEX
    for i in range(0, th):
        NG[i] = basecostFF*(finalprodrate[i]/(plantDesignCap*plantOperationalCap*daysyear/thou))/mill
        WM[i] = basecostWM*(finalprodrate[i]/(plantDesignCap*plantOperationalCap*daysyear/thou))/mill
        CC[i] = basecostCC*(finalprodrate[i]/(plantDesignCap*plantOperationalCap*daysyear/thou))/mill
        H2delivery[i] = cons2*(finalprodrate[i]*thou)/mill



    # for i in availableRate:
    #     fixedOPEX.append(dict[i])
        #fixedOPEX.append(dict[32941.26])

    # for j in finalprodrate:
    #     NG.append(basecostFF*(j/(plantDesignCap*plantOperationalCap*365/1000))/1000000);
    #     WM.append(basecostWM*(j/(plantDesignCap*plantOperationalCap*365/1000))/1000000);
    #     CC.append(basecostCC*(j/(plantDesignCap*plantOperationalCap*365/1000))/1000000);
    #     H2delivery.append(1.07*(j*1000)/1000000)
        #fixedOPEX.append(16.92)

    for i in range(0,th):
        CO2tax[i] = cons4*(ProdCO2emissions[i])/mill
        CO2TS[i] = cons5*(CO2capture[i])/mill

    # for k in ProdCO2emissions:
    #     CO2tax.append(22.43*k/1000000)

    # for l in CO2capture:
    #     CO2TS.append(15*l/1000000)

    temp1 = np.add(NG, WM);
    temp2 = np.add(CC, CO2tax)
    temp3 = np.add(CO2TS, H2delivery)
    temp4 = np.add(temp1, temp2)

    varOPEXcost = np.add(temp4, temp3) #Sum of variable OPEX costs

    OpCost = np.add(fixedOPEX, varOPEXcost); #Final Operational Costs

    #Depreciation

    # MACRSphased = (df.iloc[2:27, 10]).array
    for i in range(0, th):
        Depreciation[i] = MACRSphased[i]*(statetax+fedtax);
    # Depreciation = MACRSphased*(statetax+fedtax);

    #Salvage Value + Decommissioning
    #capInvestment = sum(df.iloc[2:27, 10].array)

    # expansionInvestment = []

    for i in range(0,th):
        expansionInvestment[0] = 0
        if expansionChange[i-1] == 1:
            expansionInvestment[i] = expansionInvestmentval
        elif expansionChange[i-1] == 0:
            expansionInvestment[i] = 0
    # expansionInvestment2 = expansionInvestment[]

    # expansionInvestment.append(0)
    # for i, v in enumerate(availableRate):
    #     if expansionChange[i] == 1:
    #         expansionInvestment.append(dict2[v])
    #     elif expansionChange[i] == 0:
    #         expansionInvestment.append(0)
    # expansionInvestment.pop()

    # capitalInvestment = initialCapCostsPhased;

    # sv = sum(expansionInvestment)+initialCapCostsPhased - sum(Depreciation)
    sv = initialCapCostsPhased + sum(expansionInvestment) - sum(Depreciation)

    #sv = df.iloc[2:27, 3].array; #capInvestment - sum(Depreciation)

    decom = pd.Series(index=nums, dtype='float64')
    for i in range(0, th):
        if i <=24:
            decom[i] = 0
        else:
            decom[i] = sv

    # decom = []
    # for i in range(24):
    #     decom.append(0)
    # decom.append(sv)



    #45Q tax credit

    # q = df.iloc[2:27, 2].array #tax credit data from csv file

    tc = pd.Series(index=nums, dtype='float64')

    for i in range(0, th):
        if CO2capture[i] >= 100000:
            tc[i] = q[i] * CO2capture[i]/mill
        else:
            tc[i] = 0

    # tc = [] #45Q tax credit

    # for i, m in enumerate(CO2capture):
    #     if m >= 100000:
    #         tc.append(q[i] * m / 1000000)
    #         #
    #     else:
    #         tc.append(0)
            
    #Cashflow

    for i in range(0, th):
        cf[i] = (revenue[i]-OpCost[i])*(1-statetax-fedtax)+Depreciation[i]+tc[i]
        dcf[i] = (cf[i] - expansionInvestment[i])/(1+discountrate)**nomyear[i]
        npv1 += dcf[i]
        npv = npv1 - initialCapCostsPhased

    # cf = (rev-OpCost)*(1-statetax-fedtax)+Depreciation+tc #Final Cashflow 
    #cf = (rev)*(1-statetax-fedtax)+Depreciation+tc+sv-decom #Final Cashflow 

    #Discounted Cashflow
    # nomyear = df.iloc[2:27, 4].array #Nominal Year Counts

    # dcf = (cf-expansionInvestment)/(1+discountrate)**nomyear #Final Discounted Cashflow 

    #npv

    # npv = sum(dcf)-initialCapCostsPhased #Final Net Present Value
    # year2 = [2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035, 2036, 2037, 2038, 2039, 2040, 2041, 2042, 2043, 2044, 2045, 2046, 2047, 2048, 2049]
    # plt.plot(year2, demandSF, label='demand')
    # plt.plot(year2, finalprodrate, label='H2 production rate')
    # plt.legend()
    # plt.title('Centralised Phased Deterministic Case Deployment Schedule')
    # plt.show()
    return -npv
# for i in range(len(cashflow)):
#     npv = npv + cashflow[i]/(1+discountrate)**i  
    
# print(f"npv of investment is {npv}");

print(test([47500, 0.5]))

# plotting graphs
# year2 = df.iloc[1:26, 0];
# plt.plot(year2, demandSF, label='demand')
# plt.plot(year2, finalprodrate, label='H2 production rate')
# plt.legend()
# plt.title('Centralised Phased Deterministic Case Deployment Schedule')
# plt.show()