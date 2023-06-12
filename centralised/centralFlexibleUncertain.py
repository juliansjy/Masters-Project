# from centralUncertainConstants import *
import pandas as pd  
import numpy as np 
from statistics import NormalDist
import random
import math
from scipy.stats import norm

#Sensitivity parameters
annVol = 0.15           #Base value 0.15
volb = 0.7              #Base value 0.7
discountrate = 0.1;     #Base value 0.1

#General constants
thou = 1000
mill = 1000000
daysyear = 365

#To calculate demand
demandyr0 = 0.0240232711111154
MDemLimit = 1.2042811301817
bsharpParam = 0.20118446680713
volyr0 = 0.5 #0.5
volM = 0.5 #0.5
scaleFactor = 1468000
BlueH2MarketShare = 0.4
SFMarketShare = 0.18

#Constants used in calculations
cons1 = 8760
cons2 = 16.92

#Operational constants
plantDesignCap = 190950;
plantOperationalCap = 0.95;
CO2emissionRate = 1.06; # Kg of CO2 per Kg of H2
CO2captureRate = 8.60; # Kg of CO2 per Kg of H2 
CO2emissionRatefeedstock =  0.28;
CO2emissionRatefuel = 1.18;
NatGasConsumption = 33411;
plantDesignCapBase = 190950;
plantDesignCapScaled = 190000;
lifetime = 25
workingHours = 8322
th = 25
dailycapreq = 190000

#Taxes
statetax = 0.0725;
fedtax = 0.21;

#Hydrogen price stuff
Hpricelow = 8
Hpricehigh = 15
Hpriceave = 11.5

#Hydrogen delivery cost stuff
H2dpave = 1.07
H2dplow = 0.749
H2dohigh = 1.391

#CO2 transport storage (ts) stuff
CO2low = 10
CO2high = 30
CO2ave = 15

#To calculate variable OPEX
basecostFF = 67836966;
basecostWM = 104434;
basecostCC = 625712;

#CO2 gas prices constants
drift = 0.00234
volatility = 0.19
CO2captradeprice = 29.15
initialCapCostsFixed = 591.73
initialCapCostsPhased = 230.53
interval = 4

#Expansion constants
iniProdRateDay = 47500
expansionThresh = 0.75
reductionDuringExpansion = 0.5
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
nomyear2 = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27]
nums = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
year = [2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035, 2036, 2037, 2038, 2039, 2040, 2041, 2042, 2043, 2044, 2045, 2046, 2047, 2048, 2049, 2050]
ngprices = [130.48,109.52,118.95,225.84,207.50,177.11,286.63,308.64,455.36,352.65,365.23,464.26,206.46,228.99,209.60,144.10,195.45,228.99,137.29,132.05,156.68,165.06,134.14,106.37,203.84]
elecprices = [7.27,6.67,6.81,6.92,6.88,6.76,6.91,7.10,6.89,6.67,6.82,6.77,6.83,6.96,6.39,6.16,5.73,5.25,5.11,4.88,5.05,4.64,4.43,4.48,4.53]

def main(designVariables):
    iniProdRateDay = designVariables[0]
    reductionDuringExpansion = designVariables[1]
    expansionThresh = designVariables[2]
    
    def npv():
        availableRate = pd.Series(index=nums, dtype='float64')
        expansionChange = pd.Series(index=nums, dtype='float64').array
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
        demandprojection = pd.Series(index=nomyear2, dtype='float64').array
        normDemProjGrowth = pd.Series(index=nums, dtype='float64')
        randomDraw = pd.Series(index=nums, dtype='float64')
        realisedGrowth = pd.Series(index=nums, dtype='float64')
        realisedNormalisedDemand = pd.Series(index=nums, dtype='float64')
        realisedDemand = pd.Series(index=nums, dtype='float64')
        demandSF = pd.Series(index=nums, dtype='float64')
        CO2prices = pd.Series(index=nums, dtype='float64')
        cf = pd.Series(index=nums, dtype='float64')
        dcf = pd.Series(index=nums, dtype='float64')
        npv1 = 0
        npv = 0

        #Start of calculations
        #Uncertainty in NPV parameters

        #Calculating Hydrogen price
        lowModeHighMode = (Hpriceave-Hpricelow)/(Hpricehigh-Hpricelow)
        randomDraw2 = random.uniform(0, 1)
        def findHprice():
            if randomDraw2 < lowModeHighMode:
                Hprice = Hpricelow+math.sqrt((Hpriceave-Hpricelow)*(Hpricehigh-Hpricelow)*randomDraw2)
            else:
                Hprice = Hpricehigh-math.sqrt((Hpricehigh-Hpricelow)*(Hpricehigh-Hpriceave)*(1-randomDraw2))
            return Hprice 

        #Calculating Hydrogen delivery price
        lowModeHighMode2 = (H2dpave-H2dplow)/(H2dohigh-H2dpave)
        def findH2deliveryprice():
            if randomDraw2 < lowModeHighMode2:
                H2dprice = H2dplow+math.sqrt((H2dpave-H2dplow)*(H2dohigh-H2dplow)*randomDraw2)
            else: 
                H2dprice = H2dohigh-math.sqrt((H2dohigh-H2dplow)*(H2dohigh-H2dpave)*(1-randomDraw2))
            return H2dprice

        #Calculating CO2 transport and storage costs
        lowModeHighMode3 = (CO2ave-CO2low)/(CO2high-CO2ave)
        def priceCO2TS():
            if randomDraw2 < lowModeHighMode3:
                CO2ts = CO2low+math.sqrt((CO2ave-CO2low)*(CO2high-CO2low)*randomDraw2)
            else:
                CO2ts = CO2high-math.sqrt((CO2high-CO2low)*(CO2high-CO2ave)*(1-randomDraw2))
            return CO2ts

        #Calculating natural gas prices
        result = [];
        for i in range(1000):
            result.append(random.choices(ngprices, weights=None, cum_weights=None, k = 25))
        samplemean = [];
        for i in result:
            samplemean.append(np.mean(i))
        totalmean = [];
        totalmean.append(np.mean(samplemean))
        totalst = []
        totalst.append(np.std(samplemean))
        ngprice = norm.ppf(random.uniform(0,1), totalmean, totalst)

        #Calculating CO2 prices
        result2 = []
        for i in range(111):
            result2.append(drift+(norm.ppf(random.uniform(0,1), 0, 1))*volatility)
        CO2tradeprices = []
        CO2tradeprices.append(CO2captradeprice)
        for i, v in enumerate(result2):
            CO2tradeprices.append(CO2tradeprices[i]*(1+result2[i]))
        result3 = [] #array used to get average price from 4 values from each year
        for i in range(11, 111):
            result3.append(CO2tradeprices[i])
        for i in range(0, th):
            CO2prices[i] = np.mean(result3[i:i+interval])

        #Demand projections
        realisedDemandyr0 = (1-volyr0)*(demandyr0)+2*demandyr0*volyr0*random.uniform(0,1)
        stoM = (1-volM)*MDemLimit+2*volM*MDemLimit*random.uniform(0,1) 
        stoa = stoM/realisedDemandyr0-1
        stob = (1-volb)*bsharpParam+2*volb*bsharpParam*random.uniform(0,1)  
        for i in range(0,len(nomyear2)):  
            demandprojection[i] = stoM/(1+stoa*np.exp(-nomyear2[i]*stob))

        for i in range(0,len(nomyear2)-1):
            normDemProjGrowth[i] = (demandprojection[i+1]-demandprojection[i])/demandprojection[i]

        for i in range(0,len(nomyear2)):
            randomDraw[i] = NormalDist(mu=0, sigma=1).inv_cdf(random.uniform(0,1))

        for i in range(0, len(nomyear2)-1):
            realisedGrowth[i] = normDemProjGrowth[i]+randomDraw[i+1]*annVol

        for i in range(0, len(nomyear2)-1):
            realisedNormalisedDemand[i] = demandprojection[i+1]*(1+realisedGrowth[i])

        for i in range(0, len(nomyear2)-1):
            realisedDemand[i] = realisedNormalisedDemand[i]*(scaleFactor*BlueH2MarketShare*SFMarketShare)

        demandSF = realisedDemand[1:th+1]
        demandSF.index = demandSF.index - 1

        #Expansion & Available Rate
        for i in range(0, th):
            availableRate[0] = (iniProdRateDay*plantOperationalCap*daysyear/thou)
            if demandSF[i] >= expansionThresh*availableRate[i] and availableRate[i] < ((iniProdRateDay*plantOperationalCap*daysyear/thou)+((((dailycapreq*plantOperationalCap*daysyear/thou)-(iniProdRateDay*plantOperationalCap*daysyear/thou))/((iniProdRateDay*plantOperationalCap*daysyear)/thou))*((iniProdRateDay*plantOperationalCap*daysyear)/thou))):
                expansionChange[i] = 1
                availableRate[i+1] = ((iniProdRateDay*plantOperationalCap*daysyear)/thou) + availableRate[i]
            else:
                expansionChange[i] = 0
                availableRate[i+1] = availableRate[i]
        availableRate2 = availableRate[0:25]

        #Final Production Rate
        for i in range(0,th):
            if expansionChange[i-1] == 1:
                finalprodrate[i] = min(demandSF[i], availableRate2[i])*(1-reductionDuringExpansion)
            elif expansionChange[i-1] == 0:
                finalprodrate[i] = min(demandSF[i], availableRate2[i])

        #Production of CO2 emission, CO2 capture, Upstream and Distribution of CO2 emissions
        for i in range(0, th):
            ProdCO2emissions[i] = CO2emissionRate*(finalprodrate[i]*thou)/thou
            CO2capture[i] = CO2captureRate*(finalprodrate[i]*thou)/thou
            upstreamCO2emissions[i] = (CO2emissionRatefeedstock/thou)*(NatGasConsumption*cons1)*(finalprodrate[i]/(plantDesignCap*plantOperationalCap*daysyear/thou))

        #Revenues
        for i in range(0,th):
            revenue[i] = findHprice()*(finalprodrate[i]*thou)/mill

        #Expansion Investment (linked to capital investment)
        for i in range(0,th):
            expansionInvestment[0] = 0
            if expansionChange[i-1] == 1:
                expansionInvestment[i] = expansionInvestmentval
            elif expansionChange[i-1] == 0:
                expansionInvestment[i] = 0

        #Total operational expenditure
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
        NG = pd.Series(index=nums, dtype='float64')
        WM = pd.Series(index=nums, dtype='float64')
        CC = pd.Series(index=nums, dtype='float64')
        CO2tax = pd.Series(index=nums, dtype='float64')
        CO2TS = pd.Series(index=nums, dtype='float64')
        H2delivery = pd.Series(index=nums, dtype='float64')
        CO2tax = pd.Series(index=nums, dtype='float64')
        CO2TS = pd.Series(index=nums, dtype='float64')

        for i in range(0,th):
            NG[i] = (NatGasConsumption*workingHours)/thou*ngprice[0]*(finalprodrate[i]/(plantDesignCapScaled*plantOperationalCap*daysyear/thou))/mill
            WM[i] = basecostWM*(finalprodrate[i]/(plantDesignCapBase*plantOperationalCap*daysyear/thou))/mill
            CC[i] = basecostCC*(finalprodrate[i]/(plantDesignCapBase*plantOperationalCap*daysyear/thou))/mill
            H2delivery[i] = findH2deliveryprice()*(finalprodrate[i]*thou)/mill

        for i in range(0,th):
            CO2tax[i] = CO2prices[i]*ProdCO2emissions[i]/mill
            CO2TS[i] = priceCO2TS()*CO2capture[i]/mill

        temp1 = np.add(NG, WM);
        temp2 = np.add(CC, CO2tax)
        temp3 = np.add(CO2TS, H2delivery)
        temp4 = np.add(temp1, temp2)
        varOPEXcost = np.add(temp4, temp3)
        OpCost = np.add(fixedOPEX, varOPEXcost); #total operational costs

        #Depreciation
        for i in range(0, th):
            Depreciation[i] = MACRSflexible[i]*(statetax+fedtax);

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
        print(npv)
        return -npv
    
    counter = 0
    for i in range(0, 2000):   #2000 runs
        counter += npv()
    enpv = counter/2000        #2000 runs
    return -enpv

print(f'ENPV value: {main([47500, 0.5, 0.75])}') #Use this to optimise ENPV
# print(f'ENPV value: {enpv([47303.2781, 0.0484, 0.6663])}') #Use this for sensitivity analysis
