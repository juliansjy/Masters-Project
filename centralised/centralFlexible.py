import pandas as pd  
import numpy as np

#General constants
thou = 1000
daysyear = 365
mill = 1000000

#To calculate demand
m = 1.2043
a = 49.1298
b = 0.2012
demand = 1468000; # raw data demand in 2050 from source. required to calc demandCAL
ms1 = 0.4
ms2 = 0.18

#Constants used in calculations
cons1 = 8760
cons2 = 1.07
cons3 = 16.92
cons4 = 22.43
cons5 = 15

#Operational constants
plantOperationalCap = 0.95;
CO2emissionRate = 1.06; # Kg of CO2 per Kg of H2
CO2captureRate = 8.60; # Kg of CO2 per Kg of H2 
CO2emissionRatefeedstock =  0.28;
CO2emissionRatefuel = 1.18;
NatGasConsumption = 33578;
plantDesignCap = 190950;
Hprice = 11.5;
iniProdRateDay = 47500
iniProdRateYear = iniProdRateDay*plantOperationalCap*365/1000
discountrate = 0.1;
time = 25;
dailycapreq = 190000
th = 25

#Taxes
statetax = 0.0725;
fedtax = 0.21;

#To calculate variable OPEX
basecostFF = 67836966;
basecostWM = 104434;
basecostCC = 625712;

#Expansion constants
expansionThresh = 0.75
reductionDuringExpansion = 0.5
initialCapCostsPhased = 266.34
expansionInvestmentval = 137.12

#Assigning different fixed OPEX depending on available rate
dict = {
    16470.63: 6.59,
    32941.26: 10.56,
    49411.89: 13.91,
    65882.52: 16.92,
}

MACRSfixed = [22,43,40,37,34,31,29,27,26,26,26,26,26,26,26,26,26,26,26,26,13,0,0,0,0]
MACRSphased = [10,19,18,16,15,19,23,21,20,25,29,28,27,31,35,34,33,32,31,31,24,18,18,18,18]
MACRSflexible = [10,19,18,16,15,14,13,12,12,17,22,21,20,25,29,33,36,35,34,33,26,19,18,18,18]
q = [46.96, 50, 54.25, 58.86, 63.86, 69.29, 75.18, 81.57, 88.51, 96.03, 104.19, 113.05, 122.66, 133.08, 144.40, 156.67, 169.99, 184.44, 200.11, 217.12, 235.58, 255.60, 277.33, 300.90, 326.48]
nomyear = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
nums = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
year = [2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035, 2036, 2037, 2038, 2039, 2040, 2041, 2042, 2043, 2044, 2045, 2046, 2047, 2048, 2049, 2050]

def main(designVariables):
    iniProdRateDay = designVariables[0]
    reductionDuringExpansion = designVariables[1]
    expansionThresh = designVariables[2]

    demandCAL = pd.Series(index=nums, dtype='float64')
    expansionChange = pd.Series(index=nums, dtype='float64').array
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
    cf = pd.Series(index=nums, dtype='float64')
    dcf = pd.Series(index=nums, dtype='float64')
    npv1 = 0
    npv = 0

    #Demand projections
    for i in range(len(year)):
        demandCAL[i] = (m/(1+a*np.exp(-b*(year[i]-year[0])))*demand)*ms1*ms2 #change p to m and in the centralConstants file
    demandSF = demandCAL[2:27]; #start from index 2 to 26 instead of 0 to 24

    #Expansion & Available Rate
    for i in range(0, th):
        availableRate[0] = iniProdRateYear
        if demandSF[i+2] >= expansionThresh*availableRate[i] and availableRate[i] < (iniProdRateYear+((((dailycapreq*plantOperationalCap*365/1000)-(iniProdRateDay*plantOperationalCap*365/1000))/((iniProdRateDay*plantOperationalCap*365)/1000))*((iniProdRateDay*plantOperationalCap*365)/1000))):
            expansionChange[i] = 1
            availableRate[i+1] = ((iniProdRateDay*plantOperationalCap*365)/1000) + availableRate[i]
        else:
            expansionChange[i] = 0
            availableRate[i+1] = availableRate[i]
    availableRate2 = availableRate[0:25]

    #Final Production Rate
    for i in range(0,th):
        if expansionChange[i-1] == 1:
            finalprodrate[i] = min(demandSF[i+2], availableRate2[i])*(1-reductionDuringExpansion)
        elif expansionChange[i-1] == 0:
            finalprodrate[i] = min(demandSF[i+2], availableRate2[i])

    #Production of CO2 emission, CO2 capture, Upstream and Distribution of CO2 emissions
    for i in range(0, th):
        ProdCO2emissions[i] = CO2emissionRate*(finalprodrate[i]*thou)/thou
        CO2capture[i] = CO2captureRate*(finalprodrate[i]*thou)/thou
        upstreamCO2emissions[i] = (CO2emissionRatefeedstock/thou)*(NatGasConsumption*cons1)*(finalprodrate[i]/(plantDesignCap*plantOperationalCap*daysyear/thou))

    #Revenues
    for i in range(0, th):
        revenue[i] = Hprice*(finalprodrate[i]*thou)/mill

    #Operational Costs
    NG = pd.Series(index=nums, dtype='float64')
    WM = pd.Series(index=nums, dtype='float64')
    CC = pd.Series(index=nums, dtype='float64')
    CO2tax = pd.Series(index=nums, dtype='float64')
    CO2TS = pd.Series(index=nums, dtype='float64')
    H2delivery = pd.Series(index=nums, dtype='float64')
    CO2tax = pd.Series(index=nums, dtype='float64')
    CO2TS = pd.Series(index=nums, dtype='float64')

    #Fixed OPEX
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

    #Variable OPEX
    for i in range(0, th):
        NG[i] = basecostFF*(finalprodrate[i]/(plantDesignCap*plantOperationalCap*daysyear/thou))/mill
        WM[i] = basecostWM*(finalprodrate[i]/(plantDesignCap*plantOperationalCap*daysyear/thou))/mill
        CC[i] = basecostCC*(finalprodrate[i]/(plantDesignCap*plantOperationalCap*daysyear/thou))/mill
        H2delivery[i] = cons2*(finalprodrate[i]*thou)/mill

    for i in range(0,th):
        CO2tax[i] = cons4*(ProdCO2emissions[i])/mill
        CO2TS[i] = cons5*(CO2capture[i])/mill

    temp1 = np.add(NG, WM);
    temp2 = np.add(CC, CO2tax)
    temp3 = np.add(CO2TS, H2delivery)
    temp4 = np.add(temp1, temp2)
    varOPEXcost = np.add(temp4, temp3) #Sum of variable OPEX costs
    OpCost = np.add(fixedOPEX, varOPEXcost); #Final Operational Costs

    #Depreciation
    for i in range(0, th):
        Depreciation[i] = MACRSflexible[i]*(statetax+fedtax);

    #Salvage Value + Decommissioning
    for i in range(0,th):
        expansionInvestment[0] = 0
        if expansionChange[i-1] == 1:
            expansionInvestment[i] = expansionInvestmentval
        elif expansionChange[i-1] == 0:
            expansionInvestment[i] = 0

    #45Q tax credit
    tc = pd.Series(index=nums, dtype='float64')

    for i in range(0, th):
        if CO2capture[i] >= 100000:
            tc[i] = q[i] * CO2capture[i]/mill
        else:
            tc[i] = 0

    #Cashflow
    for i in range(0, th):
        cf[i] = (revenue[i]-OpCost[i])*(1-statetax-fedtax)+Depreciation[i]+tc[i]
        dcf[i] = (cf[i] - expansionInvestment[i])/(1+discountrate)**nomyear[i]
        npv1 += dcf[i]
        npv = npv1 - initialCapCostsPhased
    return -npv

print(f'NPV value: {main([47500, 0.5, 0.75])}')