import matplotlib.pyplot as plt
import numpy as np

#Main function to calculate flexibility
def central(args):
    a = args[1] - args[0]
    # b = args[2] - args[0]
    # c = args[2] - args[1]
    return a

def decentral(args):
    # a = args[1] - args[0]
    b = args[2] - args[0]
    # c = args[2] - args[1]
    return b

def calc2(a, b, c, d):
    e = [a, b, c, d]
    return e

cenFlexVoF = 1051.2945 - 989.4617 #1061.1028 - 921.7072
decenFlexVoF = 798.0308 - 989.4617 #875.1929 - 630.2277

#Annual Volatility
# cenM50_1 = [896.2848, 982.1970, 1094.3626]
# cenM50_1 = [935.9597, 982.1970, 1048.7318]
# decenM50_1 = [643.5914, 743.0639, 810.1298]
mainM50_1 = [982.1970, 1094.3626, 810.1298]

# cenM25_1 = [940.8117, 998.6651, 1047.6571]
# decenM25_1 = [642.2279, 736.1674, 811.1842]
mainM25_1 = [998.6651, 1047.6571, 811.1842]

# cenP25_1 = [910.9544, 987.5118, 1036.8057]
# decenP25_1 = [636.2054, 723.3106, 789.2393]
mainP25_1 = [987.5118, 1036.8057, 789.2393]

# cenP50_1 = [886.4380, 976.6189, 1047.0834]
# decenP50_1 = [624.0570, 710.7016, 786.7276]
mainP50_1 = [976.6189, 1047.0834, 786.7276]

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
# cenM50_2 = [941.6588, 983.3856, 1030.4977]
# decenM50_2 = [643.0637, 734.6571, 811.6712]
mainM50_2 = [983.3856, 1030.4977, 811.6712]

# cenM25_2 = [937.1954, 985.9700, 1071.9906]
# decenM25_2 = [630.7510, 717.2699, 809.5360]
mainM25_2 = [985.9700, 1071.9906, 809.5360]

# cenP25_2 = [946.9252, 960.6035, 1049.7550]
# decenP25_2 = [588.3914, 726.3927, 800.5188]
mainP25_2 = [960.6035, 1049.7550, 800.5188]

# cenP50_2 = [919.0016, 944.2297, 1034.7589]
# decenP50_2 = [620.6265, 697.2473, 775.2539]
mainP50_2 = [944.2297, 1034.7589, 775.2539]

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
# cenM50_3 = [958.9159, 1033.5657, 1094.1578]
# decenM50_3 = [647.2243, 731.7966, 811.5619]
mainM50_3 = [1033.5657, 1094.1578, 811.5619]

# cenM25_3 = [963.7137, 1026.5372, 1066.5519]
# decenM25_3 = [629.6868, 717.8459, 803.0456]
mainM25_3 = [1026.5372, 1066.5519, 803.0456]

# cenP25_3 = [904.1607, 950.8312, 1057.8792]
# decenP25_3 = [608.4615, 715.4285, 781.6068]
mainP25_3 = [950.8312, 1057.8792, 781.6068]

# cenP50_3 = [867.6315, 924.1184, 1012.2813]
# decenP50_3 = [607.6411, 705.6421, 768.3788]
mainP50_3 = [924.1184, 1012.2813, 768.3788]

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
# cenM50_4 = [951.8139, 1048.5383, 1097.0584]
# decenM50_4 = [612.7190, 746.4664, 787.3803]
mainM50_4 = [1048.5383, 1097.0584, 787.3803]

# cenM25_4 = [922.8307, 1036.6916, 1049.2077]
# decenM25_4 = [607.1000, 733.3015, 781.7328]
mainM25_4 = [1036.6916, 1049.2077, 781.7328]

# cenP25_4 = [925.2086, 947.7968, 1049.6719]
# decenP25_4 = [656.7066, 689.9707, 806.4464]
mainP25_4 = [947.7968, 1049.6719, 806.4464]

# cenP50_4 = [905.8038, 942.4128, 1044.9294]
# decenP50_4 = [644.5355, 707.8535, 821.8965]
mainP50_4 = [942.4128, 1044.9294, 821.8965]

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
# cenM50_5 = [2371.2065, 2289.3679, 2412.4436]
# decenM50_5 = [1412.8215, 1463.4684, 1522.2287]
mainM50_5 = [2289.3679, 2412.4436, 1522.2287]

# cenM25_5 = [1480.2829, 1501.6779, 1643.4175]
# decenM25_5 = [924.0209, 996.0528, 1107.6907]
mainM25_5 = [1501.6779, 1643.4175, 1107.6907]

# cenP25_5 = [555.2622, 651.6729, 723.6421]
# decenP25_5 = [398.1390, 536.9045, 597.6356]
mainP25_5 = [651.6729, 723.6421, 597.6356]

# cenP50_5 = [272.7770, 418.5133, 474.0908]
# decenP50_5 = [260.2608, 360.7863, 420.2335]
mainP50_5 = [418.5133, 474.0908, 420.2335]

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
plt.title('Centralised & Decentralised Sensitivity of Annual Volatility')
plt.ylabel('Value of Flexibility')
plt.xlabel('Annual Volatility Percentage Change')
plt.legend()
plt.show()

title = ["25%", "37.5%", "50%","62.5%", "75%"]
plt.plot(title, cenVolatilityYear, label="Centralised")
plt.plot(title, decenVolatilityYear, label="Decentralised")
plt.title('Centralised & Decentralised Sensitivity of Volatility in Year 0')
plt.ylabel('Value of Flexibility')
plt.xlabel('Volatility in Year 0 Percentage Change')
plt.legend()
plt.show()

title = ["25%", "37.5%", "50%","62.5%", "75%"]
plt.plot(title, cenVolatilityM, label="Centralised")
plt.plot(title, decenVolatilityM, label="Decentralised")
plt.title('Centralised & Decentralised Sensitivity of Volatility M')
plt.ylabel('Value of Flexibility')
plt.xlabel('Volatility M Percentage Change')
plt.legend()
plt.show()

title = ["35%", "52.5%", "70%", "87.5%", "105%"]
plt.plot(title, cenVolatilityB, label="Centralised")
plt.plot(title, decenVolatilityB, label="Decentralised")
plt.title('Centralised & Decentralised Sensitivity of Volatility B')
plt.ylabel('Value of Flexibility')
plt.xlabel('Volatility B Percentage Change')
plt.legend()
plt.show()

title = ["5%", "7.5%", "10%", "12.5%", "15%"]
plt.plot(title, cenVolatilityDiscountRate, label="Centralised")
plt.plot(title, decenVolatilityDiscountRate, label="Decentralised")
plt.title('Centralised & Decentralised Sensitivity of Volatility in Discount Rate')
plt.ylabel('Value of Flexibility')
plt.xlabel('Volatility in Discount Rate Percentage Change')
plt.legend()
plt.show()