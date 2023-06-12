import matplotlib.pyplot as plt
import numpy as np

#Main function to calculate flexibility
def central(args):
    a = args[1] - args[0]
    return a

def decentral(args):
    b = args[2] - args[0]
    return b

def calc2(a, b, c, d):
    e = [a, b, c, d]
    return e

cenFlexVoF = 1294.7958-1124.6001  #1061.1028 - 921.7072 1051.2945 - 989.4617
decenFlexVoF = 1135.6080-1124.6001 #875.1929 - 630.2277 798.0308 - 989.4617

#Annual Volatility
mainM50_1 = [1089.3222, 1233.9550, 1110.0984] #[982.1970, 1094.3626, 810.1298] 
mainM25_1 = [1066.9065, 1244.6756, 1094.3071] #[998.6651, 1047.6571, 811.1842]
mainP25_1 = [1088.2814, 1220.4914, 1105.4884] #[987.5118, 1036.8057, 789.2393]
mainP50_1 = [1081.1006, 1193.0606, 1097.2009] #[976.6189, 1047.0834, 786.7276]

cenM50vof_1 = central(mainM50_1)
decenM50vof_1 = decentral(mainM50_1)
cenM25vof_1 = central(mainM25_1)
decenM25vof_1 = decentral(mainM25_1)
cenP25vof_1 = central(mainP25_1)
decenP25vof_1 = decentral(mainP25_1)
cenP50vof_1 = central(mainP50_1)
decenP50vof_1 = decentral(mainP50_1)

cenAnnualVolatility = calc2(cenM50vof_1, cenM25vof_1, cenP25vof_1, cenP50vof_1)
cenAnnualVolatility.insert(2, cenFlexVoF)
decenAnnualVolatility = calc2(decenM50vof_1, decenM25vof_1, decenP25vof_1, decenP50vof_1)
decenAnnualVolatility.insert(2, decenFlexVoF)

#Volatility Year 0
mainM50_2 = [1138.0563, 1280.5790, 1134.3588] #[983.3856, 1030.4977, 811.6712]
mainM25_2 = [1097.6681, 1303.2662, 1137.7909] #[985.9700, 1071.9906, 809.5360]
mainP25_2 = [1059.7499, 1200.1682, 1098.6533] #[960.6035, 1049.7550, 800.5188]
mainP50_2 = [1076.7213, 1226.1826, 1101.9709] #[944.2297, 1034.7589, 775.2539]

cenM50vof_2 = central(mainM50_2)
decenM50vof_2 = decentral(mainM50_2)
cenM25vof_2 = central(mainM25_2)
decenM25vof_2 = decentral(mainM25_2)
cenP25vof_2 = central(mainP25_2)
decenP25vof_2 = decentral(mainP25_2)
cenP50vof_2 = central(mainP50_2)
decenP50vof_2 = decentral(mainP50_2)

cenVolatilityYear = calc2(cenM50vof_2, cenM25vof_2, cenP25vof_2, cenP50vof_2)
cenVolatilityYear.insert(2, cenFlexVoF)
decenVolatilityYear = calc2(decenM50vof_2, decenM25vof_2, decenP25vof_2, decenP50vof_2)
decenVolatilityYear.insert(2, decenFlexVoF)

# #Volatility M
mainM50_3 = [1120.4874, 1269.1022, 1135.3497]
mainM25_3 = [1073.7870, 1249.1250, 1117.5388]
mainP25_3 = [1072.7311, 1233.3062, 1082.5242]
mainP50_3 = [990.7980, 1170.1455, 1073.4901]

cenM50vof_3 = central(mainM50_3)
decenM50vof_3 = decentral(mainM50_3)
cenM25vof_3 = central(mainM25_3)
decenM25vof_3 = decentral(mainM25_3)
cenP25vof_3 = central(mainP25_3)
decenP25vof_3 = decentral(mainP25_3)
cenP50vof_3 = central(mainP50_3)
decenP50vof_3 = decentral(mainP50_3)

cenVolatilityM = calc2(cenM50vof_3, cenM25vof_3, cenP25vof_3, cenP50vof_3)
cenVolatilityM.insert(2, cenFlexVoF)
decenVolatilityM = calc2(decenM50vof_3, decenM25vof_3, decenP25vof_3, decenP50vof_3)
decenVolatilityM.insert(2, decenFlexVoF)

# #Volatility B
mainM50_4 = [1150.1112, 1254.3232, 1117.0693] #[1048.5383, 1097.0584, 787.3803]
mainM25_4 = [1109.7422, 1235.2528, 1085.0702] #[1036.6916, 1049.2077, 781.7328]
mainP25_4 = [1007.8645, 1237.6247, 1114.9151] #[947.7968, 1049.6719, 806.4464]
mainP50_4 = [1029.5183, 1297.9006, 1109.9649] #[942.4128, 1044.9294, 821.8965]

cenM50vof_4 = central(mainM50_4)
decenM50vof_4 = decentral(mainM50_4)
cenM25vof_4 = central(mainM25_4)
decenM25vof_4 = decentral(mainM25_4)
cenP25vof_4 = central(mainP25_4)
decenP25vof_4 = decentral(mainP25_4)
cenP50vof_4 = central(mainP50_4)
decenP50vof_4 = decentral(mainP50_4)

cenVolatilityB = calc2(cenM50vof_4, cenM25vof_4, cenP25vof_4, cenP50vof_4)
cenVolatilityB.insert(2, cenFlexVoF)
decenVolatilityB = calc2(decenM50vof_4, decenM25vof_4, decenP25vof_4, decenP50vof_4)
decenVolatilityB.insert(2, decenFlexVoF)

# #Discount Rate
mainM50_5 = [2576.8265, 2735.7860, 2032.8422] #[2289.3679, 2412.4436, 1522.2287]
mainM25_5 = [1654.6167, 1817.2609, 1501.8914] #[1501.6779, 1643.4175, 1107.6907]
mainP25_5 = [717.2100, 833.5633, 849.1241] #[651.6729, 723.6421, 597.6356]
mainP50_5 = [507.0322, 575.4193, 654.9336] #[418.5133, 474.0908, 420.2335]

cenM50vof_5 = central(mainM50_5)
decenM50vof_5 = decentral(mainM50_5)
cenM25vof_5 = central(mainM25_5)
decenM25vof_5 = decentral(mainM25_5)
cenP25vof_5 = central(mainP25_5)
decenP25vof_5 = decentral(mainP25_5)
cenP50vof_5 = central(mainP50_5)
decenP50vof_5 = decentral(mainP50_5)

cenVolatilityDiscountRate = (calc2(cenM50vof_5, cenM25vof_5, cenP25vof_5, cenP50vof_5))
cenVolatilityDiscountRate.insert(2, cenFlexVoF)
decenVolatilityDiscountRate = (calc2(decenM50vof_5, decenM25vof_5, decenP25vof_5, decenP50vof_5))
decenVolatilityDiscountRate.insert(2, decenFlexVoF)

print(cenAnnualVolatility)
print(decenAnnualVolatility)
print(cenVolatilityYear)
print(decenVolatilityYear)
print(cenVolatilityM)
print(decenVolatilityM)
print(cenVolatilityB)
print(decenVolatilityB)
print(cenVolatilityDiscountRate)
print(decenVolatilityDiscountRate)

#Graphs
title = ["7.5%", "11.25%", "15%", "18.75%", "22.5%"]
plt.plot(title, cenAnnualVolatility, label="Centralised")
plt.plot(title, decenAnnualVolatility, label="Decentralised")
plt.title('Centralised & Decentralised Sensitivity to Annual Volatility')
plt.ylabel('Value of Flexibility')
plt.xlabel('Annual Volatility')
plt.legend()
plt.show()

title = ["25%", "37.5%", "50%","62.5%", "75%"]
plt.plot(title, cenVolatilityYear, label="Centralised")
plt.plot(title, decenVolatilityYear, label="Decentralised")
plt.title('Centralised & Decentralised Sensitivity to Volatility in Year 0')
plt.ylabel('Value of Flexibility')
plt.xlabel('Volatility in Year 0')
plt.legend()
plt.show()

title = ["25%", "37.5%", "50%","62.5%", "75%"]
plt.plot(title, cenVolatilityM, label="Centralised")
plt.plot(title, decenVolatilityM, label="Decentralised")
plt.title('Centralised & Decentralised Sensitivity to Volatility M')
plt.ylabel('Value of Flexibility')
plt.xlabel('Volatility M')
plt.legend()
plt.show()

title = ["35%", "52.5%", "70%", "87.5%", "105%"]
plt.plot(title, cenVolatilityB, label="Centralised")
plt.plot(title, decenVolatilityB, label="Decentralised")
plt.title('Centralised & Decentralised Sensitivity to Volatility B')
plt.ylabel('Value of Flexibility')
plt.xlabel('Volatility B')
plt.legend()
plt.show()

title = ["5%", "7.5%", "10%", "12.5%", "15%"]
plt.plot(title, cenVolatilityDiscountRate, label="Centralised")
plt.plot(title, decenVolatilityDiscountRate, label="Decentralised")
plt.title('Centralised & Decentralised Sensitivity to Discount Rate')
plt.ylabel('Value of Flexibility')
plt.xlabel('Discount Rate')
plt.legend()
plt.show()