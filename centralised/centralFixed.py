import pandas as pd; 
import numpy as np;

#General constants
thou = 1000         #thousand
daysyear = 365      #days in a year
mill = 1000000      #million

#To calculate demand
m = 1.2043
a = 49.1298
b = 0.2012
demand = 1468000;   #Raw data demand in 2050 from source. required to calc demandCAL
ms1 = 0.4
ms2 = 0.18

#Constants used in calculations
cons1 = 8760        #calculates upstream CO2 emissions
cons2 = 1.07        #calculates H2 delivery costs
cons3 = 16.92       #calculates fixedOPEX
cons4 = 22.43       #calculates CO2 tax
cons5 = 15          #calculates CO2 transport and storage costs

#Operational constants
discountrate = 0.1; #discount rate base value 10% 
plantOperationalCap = 0.95;             
CO2emissionRate = 1.06;                 # Kg of CO2 per Kg of H2
CO2captureRate = 8.60;                  # Kg of CO2 per Kg of H2 
CO2emissionRatefeedstock =  0.28;
NatGasConsumption = 33578;
plantDesignCap = 190950;
dailyprodrate = 190000;                 #Daily production rate
Hprice = 11.5;                          #Hydrogen price

#Taxes
statetax = 0.0725;      #state tax
fedtax = 0.21;          #federal tax

#To calculate Variable OPEX
basecostFF = 67836966;
basecostWM = 104434;
basecostCC = 625712;

#Initial capital costs
initialCapCostsFixed = 591.73

#time horizon
th = 25     #time horizon

MACRSfixed = [22,43,40,37,34,31,29,27,26,26,26,26,26,26,26,26,26,26,26,26,13,0,0,0,0]
MACRSphased = [10,19,18,16,15,19,23,21,20,25,29,28,27,31,35,34,33,32,31,31,24,18,18,18,18]
MACRSflexible = [10,19,18,16,15,14,13,12,12,17,22,21,20,25,29,33,36,35,34,33,26,19,18,18,18]
q = [46.96, 50, 54.25, 58.86, 63.86, 69.29, 75.18, 81.57, 88.51, 96.03, 104.19, 113.05, 122.66, 133.08, 144.40, 156.67, 169.99, 184.44, 200.11, 217.12, 235.58, 255.60, 277.33, 300.90, 326.48]
nomyear = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]

def main(designVariables): 
    dailyprodrate = designVariables[0]

    nums = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
    demandCAL = pd.Series(index=nums, dtype='float64')
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
    sv = 0
    cf = pd.Series(index=nums, dtype='float64')
    dcf = pd.Series(index=nums, dtype='float64')
    npv1 = 0
    npv = 0

    #Start of calcs
    #Demand for all 25 years in time horizon
    year = [2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035, 2036, 2037, 2038, 2039, 2040, 2041, 2042, 2043, 2044, 2045, 2046, 2047, 2048, 2049, 2050]
    for i in range(len(year)):
        demandCAL[i] = (m/(1+a*np.exp(-b*(year[i]-year[0])))*demand)*ms1*ms2
    demandSF = demandCAL[2:27];                                                             #start from index 2 to 26 instead of 0 to 24

    #Available rate
    for i in range(0, th):
        availableRate[i] = (dailyprodrate * plantOperationalCap * (daysyear/thou))

    #Final Production Rate
    for i in range(0, th):
        finalprodrate[i] = min(demandSF[i+2], availableRate[i])

    #Production of CO2 emission, CO2 capture, Upstream and Distribution of CO2 emissions
    for i in range(0, th):
        ProdCO2emissions[i] = CO2emissionRate*(finalprodrate[i]*thou)/thou
        CO2capture[i] = CO2captureRate*(finalprodrate[i]*thou)/thou
        upstreamCO2emissions[i] = (CO2emissionRatefeedstock/thou)*(NatGasConsumption*cons1)*(finalprodrate[i]/(plantDesignCap*plantOperationalCap*daysyear/thou))

    #Revenues
    for i in range(0, th):
        revenue[i] = Hprice*(finalprodrate[i]*thou)/mill

    #Total operational expenditure
    NG = pd.Series(index=nums, dtype='float64')
    WM = pd.Series(index=nums, dtype='float64')
    CC = pd.Series(index=nums, dtype='float64')
    CO2tax = pd.Series(index=nums, dtype='float64')
    CO2TS = pd.Series(index=nums, dtype='float64')
    H2delivery = pd.Series(index=nums, dtype='float64')
    CO2tax = pd.Series(index=nums, dtype='float64')
    CO2TS = pd.Series(index=nums, dtype='float64')

    for i in range(0, th):
        NG[i] = basecostFF*(finalprodrate[i]/(plantDesignCap*plantOperationalCap*daysyear/thou))/mill       #Natural gas costs
        WM[i] = basecostWM*(finalprodrate[i]/(plantDesignCap*plantOperationalCap*daysyear/thou))/mill       #Water makeup costs
        CC[i] = basecostCC*(finalprodrate[i]/(plantDesignCap*plantOperationalCap*daysyear/thou))/mill       #Carbon capture costs
        H2delivery[i] = cons2*(finalprodrate[i]*thou)/mill                  #H2 delivery costs
        fixedOPEX[i] = cons3                    #Fixed Operational costs

    for i in range(0,th):
        CO2tax[i] = cons4*(ProdCO2emissions[i])/mill        #CO2 tax
        CO2TS[i] = cons5*(CO2capture[i])/mill               #CO2 transport and storage costs

    temp1 = np.add(NG, WM)
    temp2 = np.add(CC, CO2tax)
    temp3 = np.add(CO2TS, H2delivery)
    temp4 = np.add(temp1, temp2)
    varOPEXcost = np.add(temp4, temp3)
    OpCost = np.add(fixedOPEX, varOPEXcost);    #Total operational cost

    #Depreciation
    for i in range(0, th):
        Depreciation[i] = MACRSfixed[i]*(statetax+fedtax);

    #45Q tax credit
    tc = pd.Series(index=nums, dtype='float64')
    for i in range(0, th):
        if CO2capture[i] >= 100000:
            tc[i] = q[i] * CO2capture[i]/mill
        else:
            tc[i] = 0

    #Final Cashflow and NPV
    for i in range(0, th):
        cf[i] = (revenue[i]-OpCost[i])*(1-statetax-fedtax)+Depreciation[i]+tc[i]
        dcf[i] = cf[i]/(1+discountrate)**nomyear[i]
        npv1 += dcf[i]
        npv = npv1 - initialCapCostsFixed
    return -npv

print(f'NPV value: {main([190000])}')

