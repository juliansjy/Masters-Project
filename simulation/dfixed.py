import pandas as pd  
import numpy as np 
from statistics import NormalDist
import random
import math
from scipy.stats import norm
import matplotlib as plt

#General constants
thou = 1000
mill = 1000000
daysyear = 365

#To calculate demand
demandyr0 = 0.0240232711111154
MDemLimit = 1.2042811301817
aTransParam = 49.1297731108936
bsharpParam = 0.20118446680713
annVol = 0.15
volyr0 = 0.5
volM = 0.5
volb = 0.7
scaleFactor = 1468000
BlueH2MarketShare = 0.4
SFMarketShare = 0.18

#Constants used in calculations
cons1 = 8760
cons2 = 16.92

#Operational constants
CO2emissionRate = 1.06; # Kg of CO2 per Kg of H2
endcapreq = 65882.50;
singlemodprodrate = 470.68
plantfixedoperationalcosts = 1.09
plantextensioncost = 0.54
discountrate = 0.1;
th = 25
modsbuilt = 140
plantsbuilt = 35
plantannualprodrate = 1883
modcapex = 1.57
setupcost = 3.24
othervariableoperating = 1800;
othervariableoperating2 = 5.32;
ngusage = 0.155797012;
ngusageannual = 85963552.88;
elecusage = 1.11;
industrialelec = 0.061
waterusage = 5.77;
processwater = 0.0024;
btutokwh = 293.07;
ngtobtu = 58.36;

#Taxes
statetax = 0.0725;
fedtax = 0.21;

#Hydrogen price
Hpricelow = 8
Hpricehigh = 15
Hpriceave = 11.5

#Hydrogen delivery cost 
H2dpave = 1.07
H2dplow = 0.749
H2dohigh = 1.391

#CO2 emissions
CO2emissionsCH4 = 0.185
CO2emissionrate = 15903257.28
CO2captureeff = 0.9
CO2captured = 14312931.55
annualCO2emissions = 1590325.73

#CO2 transport storage (ts)
CO2low = 10
CO2high = 30
CO2ave = 15

#CO2 storage and compression (sc)
CO2schigh = 289
CO2sclow = 155
CO2scave = 222

#CO2 gas prices constants
drift = 0.00234
volatility = 0.19
CO2captradeprice = 29.15
initialCapCostsFixed = 591.73
initialCapCostsPhased = 230.53
interval = 4

#To calculate variable OPEX
basecostFF = 67836966;
basecostWM = 104434;
basecostCC = 625712;

MACRSfixed = [22,43,40,37,34,31,29,27,26,26,26,26,26,26,26,26,26,26,26,26,13,0,0,0,0]
MACRSphased = [10,19,18,16,15,19,23,21,20,25,29,28,27,31,35,34,33,32,31,31,24,18,18,18,18]
MACRSflexible = [10,19,18,16,15,14,13,12,12,17,22,21,20,25,29,33,36,35,34,33,26,19,18,18,18]
q = [46.96, 50, 54.25, 58.86, 63.86, 69.29, 75.18, 81.57, 88.51, 96.03, 104.19, 113.05, 122.66, 133.08, 144.40, 156.67, 169.99, 184.44, 200.11, 217.12, 235.58, 255.60, 277.33, 300.90, 326.48]
nomyear = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
nomyear2 = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27]
nums = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
year = [2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035, 2036, 2037, 2038, 2039, 2040, 2041, 2042, 2043, 2044, 2045, 2046, 2047, 2048, 2049, 2050]
ngprices = [130.48,109.52,118.95,225.84,207.50,177.11,286.63,308.64,455.36,352.65,365.23,464.26,206.46,228.99,209.60,144.10,195.45,228.99,137.29,132.05,156.68,165.06,134.14,106.37,203.84]
elecprices = [0.0727,0.0667,0.0681,0.0692,0.0688,0.0676,0.0691,0.0710,0.0689,0.0667,0.0682,0.0677,0.0683,0.0696,0.0639,0.0616,0.0573,0.0525,0.0511,0.0488,0.0505,0.0464,0.0443,0.0448,0.0453]

def npv():
    availableRate = pd.Series(index=nums, dtype='float64')
    dsbalance = pd.Series(index=nums, dtype='float64')
    H2prodrate = pd.Series(index=nums, dtype='float64')
    CO2emissions = pd.Series(index=nums, dtype='float64')
    CO2capture = pd.Series(index=nums, dtype='float64')
    revenue = pd.Series(index=nums, dtype='float64')
    fixedcost = pd.Series(index=nums, dtype='float64')
    varcosts = pd.Series(index=nums, dtype='float64')
    opcosts = pd.Series(index=nums, dtype='float64')
    Depreciation = pd.Series(index=nums, dtype='float64')
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

    #Calculating CO2 transport and storage costs
    lowModeHighMode3 = (CO2ave-CO2low)/(CO2high-CO2ave)
    def priceCO2TS():
        if randomDraw2 < lowModeHighMode3:
            CO2ts = CO2low+math.sqrt((CO2ave-CO2low)*(CO2high-CO2low)*randomDraw2)
        else:
            CO2ts = CO2high-math.sqrt((CO2high-CO2low)*(CO2high-CO2ave)*(1-randomDraw2))
        return CO2ts

    #Calculating CO2 storage and compression costs
    lowModeHighMode4 = (CO2scave-CO2sclow)/(CO2schigh-CO2scave)
    def priceCO2SC():
        if randomDraw2 < lowModeHighMode4:
            CO2sc = CO2sclow+math.sqrt((CO2scave-CO2sclow)*(CO2schigh-CO2sclow)*randomDraw2)
        else:
            CO2sc = CO2schigh-math.sqrt((CO2schigh-CO2sclow)*(CO2schigh-CO2scave)*(1-randomDraw2))
        return CO2sc
    
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

    #Calculating Electricity price
    result = [];
    for i in range(1000):
        result.append(random.choices(elecprices, weights=None, cum_weights=None, k = 25))
    samplemean = [];
    for i in result:
        samplemean.append(np.mean(i))
    totalmean = [];
    totalmean.append(np.mean(samplemean))
    totalst = []
    totalst.append(np.std(samplemean))
    elecprice = norm.ppf(random.uniform(0,1), totalmean, totalst)

    #Calculating CO2 price
    result2 = []
    for i in range(111): #108
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

    #Available rate
    for i in range(0, th):
        availableRate[i] = singlemodprodrate*(math.ceil(endcapreq/singlemodprodrate))

    #Demand and supply balance per year
    for i in range(0,th):
        dsbalance[i] = min(demandSF[i], endcapreq)-availableRate[i]

    #Hydrogen production rate
    for i in range(0,th):
        H2prodrate[i] = min(demandSF[i], availableRate[i])

    #CO2 emissions and capture 
    for i in range(0,th):
        CO2emissions[i] = (ngusage*btutokwh*CO2emissionsCH4)*(1-CO2captureeff)*(H2prodrate[i]*thou)/thou
        CO2capture[i] = (ngusage*btutokwh*CO2emissionsCH4)*(CO2captureeff)*(CO2emissions[i]*thou)/thou

    #Revenues
    for i in range(0,th):
        revenue[i] = findHprice()*(H2prodrate[i]*thou)/mill

    #Total capital investment
    modulecapex = modcapex * modsbuilt
    plantcapex = (math.ceil(modsbuilt/(plantannualprodrate/singlemodprodrate))) * setupcost
    tci = modulecapex+plantcapex

    #Total operational expenditure
    #Fixed OPEX
    for i in range(0,th):
        if nomyear[i] == 20:
            fixedcost[i] = (plantfixedoperationalcosts+plantextensioncost)*plantsbuilt
        else:
            fixedcost[i] = plantfixedoperationalcosts*plantsbuilt

    #Variable OPEX
    ng = pd.Series(index=nums, dtype='float64'); #natural gas
    elec = pd.Series(index=nums, dtype='float64');  #electricity
    ts = pd.Series(index=nums, dtype='float64');    #transport and storage
    pw = pd.Series(index=nums, dtype='float64');    #process water
    tax = pd.Series(index=nums, dtype='float64');   #CO2 tax
    cc = pd.Series(index=nums, dtype='float64');    #CO2 capture and compression
    ovoc = pd.Series(index=nums, dtype='float64');  #other variable operating costs

    for i in range(0,th):
        ng[i] = ngusage/ngtobtu*ngprice[0]*(H2prodrate[i]*thou)/mill
        elec[i] = elecusage*(H2prodrate[i]*thou)*elecprice[0]/mill
        pw[i] = waterusage*processwater*(H2prodrate[i]*thou)/mill
        ovoc[i] = (othervariableoperating*othervariableoperating2)*plantsbuilt/mill

    for i in range(0,th):
        ts[i] = CO2capture[i]*priceCO2TS()/mill
        cc[i] = priceCO2SC()*CO2capture[i]/mill
        tax[i] = CO2prices[i]*CO2emissions[i]/mill

    temp1 = np.add(ng, elec)
    temp2 = np.add(ts, pw)
    temp3 = np.add(tax, cc)
    temp4 = np.add(temp1, temp2)
    temp5 = np.add(temp3, ovoc)
    varcosts = np.add(temp4, temp5)
    opcosts = np.add(fixedcost, varcosts)

    #Depreciation
    for i in range(0, th):
        Depreciation[i] = MACRSfixed[i]*(statetax+fedtax);

    #Cashflow
    for i in range(0, th):
        cf[i] = (revenue[i]-opcosts[i])*(1-statetax-fedtax)+Depreciation[i]
        dcf[i] = cf[i]/(1+discountrate)**nomyear[i]
    npv1 = sum(dcf[:-5])
    npv = npv1 - tci
    print(npv)
    return npv

kiwi = []                                                       #array holding all 2000 npv values
counter = 0                                                     #counter used to calculate the average enpv over 2000 runs
for i in range(0, 2000):   #2000 runs
    counter += npv()
    kiwi.append(npv())
enpv4 = counter/2000        #2000 runs                                              

maximumenpv = max(kiwi)                                         #maximum value in kiwi
minimumenpv = min(kiwi)                                         #minimum values in kiwi

bounds4 = []
for i in range(0, 20):
    calc = minimumenpv+(maximumenpv-minimumenpv)/20*i
    bounds4.append(calc)

totalCount = []
for i in range(len(bounds4)):
    counter2 = 0
    for j in range(len(kiwi)):
        if kiwi[j] <= bounds4[i]:
            counter2 += 1
    totalCount.append(counter2)

cdf4 = []
for i in range(len(totalCount)):
    calc = (totalCount[i]/len(kiwi))*100
    cdf4.append(calc)

#VaR VaG
var = np.percentile(kiwi, 10)
print(f"value at risk (10%): {var}")
vag = np.percentile(kiwi, 90)
print(f"value at gain (90%): {vag}")

#Histogram creation
histogramCount = []
histogramCount.append(totalCount[0])
for i in range(1, 20): 
    calc = totalCount[i] - totalCount[i-1]
    histogramCount.append(calc)

width = 50
plt.bar(bounds4, histogramCount, width=width)
plt.ylabel("Frequency")
plt.xlabel("NPV bins")
plt.title('Decentralised Fixed Plant NPV Distribution')
plt.show()

print(f'cdf4 = {cdf4}')
print(len(cdf4))

plt.plot(bounds4, cdf4)
plt.ylabel("Cumulative Distribution Function")
plt.xlabel("NPV")
plt.show()