import pandas as pd; 
import numpy as np;
import math

#General constants
thou = 1000
mill = 1000000
daysyear = 365

#To calculate demand
m = 1.20428113;
a = 49.12977311;
b = 0.201184467;
demand = 1468000;
ms1 = 0.4
ms2 = 0.18

#Constants used in calculations
cons1 = 8760
cons2 = 1.07
cons3 = 16.92
cons4 = 22.43
cons5 = 15

#Operational constants
Hprice = 11.5;
modcapex = 1.57
setupcost = 3.24
plantfixedoperationalcosts = 1.09
plantextensioncost = 0.54
modsbuilt = 140
plantsbuilt = 35
CO2price = 22.43
ngprice = 4.16
discountrate = 0.1; # Nine percent per annum
time = 25;
depRate = 0.04
th = 25
singleModProdRate = 470.68
endcapreq = 65882.50;
plantannualprodrate = 1883
ngusage = 0.155797012;
elecusage = 1.11;
industrialelec = 0.061
waterusage = 5.77;
btutokwh = 293.07;
CO2emissionsCH4 = 0.185
CO2captureeff = 0.9
processwater = 0.0024;
CO2ts = 15;
capturecosts = 222;
othervariableoperating = 1800;
othervariableoperating2 = 5.32;

#Taxes
statetax = 0.0725;
fedtax = 0.21;

MACRSfixed = [12.52,24.11,22.31,20.64,19.07,17.67,16.33,15.10,14.90,14.90,14.90,14.90,14.90,14.90,14.90,14.90,14.90,14.90,14.90,14.90,7.48,0.00,0.00,0.00,0.00]
MACRSphased = [3.16,6.09,5.63,5.21,4.81,7.62,10.21,9.44,8.97,11.74,14.31,13.51,12.78,15.37,17.83,17.06,16.34,15.91,15.57,15.24,13.07,11.14,11.14,11.14,11.14]
MACRSflexible = [3.16,6.09,5.63,5.21,4.81,4.46,4.12,3.81,3.76,4.89,5.94,7.09,10.78,12.87,12.19,14.42,16.48,17.05,17.58,16.74,14.31,11.94,11.50,11.30,11.16]
q = [46.96, 50, 54.25, 58.86, 63.86, 69.29, 75.18, 81.57, 88.51, 96.03, 104.19, 113.05, 122.66, 133.08, 144.40, 156.67, 169.99, 184.44, 200.11, 217.12, 235.58, 255.60, 277.33, 300.90, 326.48]
nomyear = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
year = [2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035, 2036, 2037, 2038, 2039, 2040, 2041, 2042, 2043, 2044, 2045, 2046, 2047, 2048, 2049, 2050]
nums = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
elecprices = [7.27,6.67,6.81,6.92,6.88,6.76,6.91,7.10,6.89,6.67,6.82,6.77,6.83,6.96,6.39,6.16,5.73,5.25,5.11,4.88,5.05,4.64,4.43,4.48,4.53]

def main(designVariables):
    singleModProdRate = designVariables[0]

    demandCAL = pd.Series(index=nums, dtype='float64')
    availableRate = pd.Series(index=nums, dtype='float64')
    dsbalance = pd.Series(index=nums, dtype='float64')
    H2prodrate = pd.Series(index=nums, dtype='float64')
    CO2emissions = pd.Series(index=nums, dtype='float64')
    CO2capture = pd.Series(index=nums, dtype='float64')
    CO2capture = pd.Series(index=nums, dtype='float64')
    revenue = pd.Series(index=nums, dtype='float64')
    fixedcost = pd.Series(index=nums, dtype='float64')
    Depreciation = pd.Series(index=nums, dtype='float64')
    cf = pd.Series(index=nums, dtype='float64')
    dcf = pd.Series(index=nums, dtype='float64')
    npv1 = 0
    npv = 0

    #Demand projections
    for i in range(len(year)):
        demandCAL[i] = (m/(1+a*np.exp(-b*(year[i]-year[0])))*demand)*ms1*ms2 
    demandSF = demandCAL[2:27]; #start from index 2 to 26 instead of 0 to 24

    #Available rate
    for i in range(0, th):
        availableRate[i] = singleModProdRate*math.ceil(endcapreq/singleModProdRate)

    #Demand and supply balance
    for i in range(0,th):
        dsbalance[i] = min(demandSF[i+2], endcapreq)-availableRate[i]

    #Hydrogen production rate
    for i in range(0,th):
        H2prodrate[i] = min(demandSF[i+2], availableRate[i])

    #CO2 emissions and capture
    for i in range(0,th):
        CO2emissions[i] = (ngusage*btutokwh*CO2emissionsCH4)*(1-CO2captureeff)*(H2prodrate[i]*thou)/thou
        CO2capture[i] = (ngusage*btutokwh*CO2emissionsCH4)*(CO2captureeff)*(CO2emissions[i]*thou)/thou

    #Revenues
    for i in range(0,th):
        revenue[i] = Hprice*(H2prodrate[i]*thou)/mill

    #Total capital investment
    modulecapex = modcapex * modsbuilt
    plantcapex = (math.ceil(modsbuilt/(plantannualprodrate/singleModProdRate))) * setupcost
    tci = modulecapex+plantcapex

    #Total operational expenditure
    
    #Fixed OPEX
    for i in range(0,th):
        if nomyear[i] == 20:
            fixedcost[i] = (plantfixedoperationalcosts+plantextensioncost)*plantsbuilt
        else:
            fixedcost[i] = plantfixedoperationalcosts*plantsbuilt

    #Variable OPEX
    ng = pd.Series(index=nums, dtype='float64');    #natural gas
    elec = pd.Series(index=nums, dtype='float64');  #electricity
    ts = pd.Series(index=nums, dtype='float64');    #transport and storage
    pw = pd.Series(index=nums, dtype='float64');    #process water
    tax = pd.Series(index=nums, dtype='float64');   #CO2 tax
    cc = pd.Series(index=nums, dtype='float64');    #CO2 capture and compression
    ovoc = pd.Series(index=nums, dtype='float64');  #other variable operating costs

    for i in range(0,th):
        ng[i] = ngusage*ngprice*(H2prodrate[i]*thou)/mill
        elec[i] = elecusage*(H2prodrate[i]*thou)*industrialelec/mill
        pw[i] = waterusage*processwater*(H2prodrate[i]*thou)/mill
        ovoc[i] = (othervariableoperating*othervariableoperating2)*plantsbuilt/mill

    for i in range(0,th):
        ts[i] = CO2ts*CO2capture[i]/mill
        cc[i] = capturecosts*CO2capture[i]/mill
        tax[i] = CO2price*CO2emissions[i]/mill

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
    return -npv

print(f'NPV value: {main([470.68])}')