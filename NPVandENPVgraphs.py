import matplotlib.pyplot as plt

#bar graphs

#Centralised Bar Graph - Certain

title = ['Fixed', 'Phased', 'Flexible']
a = [879.4503, 957.2670, 989.3178] #Unopti
b = [903.9705, 1049.6555, 1182.6733] #Opti
width = 0.6
plt.bar(title, b, label='Optimised', width=width)
plt.bar(title, a, label='Unoptimised', width=width)
plt.ylabel('NPV')
plt.title('Centralised NPV Values - Certain')
plt.legend()
plt.show()

#Centralised Bar Graph - Uncertain

# title = ['Fixed', 'Phased', 'Flexible']
# a = [946.0850, 989.4617, 1051.2945] #Unopti
# b = [990.8074, 1124.6001, 1294.7958] #Opti
# width = 0.6
# plt.bar(title, b, label='Optimised', width=width)
# plt.bar(title, a, label='Unoptimised', width=width)
# plt.ylabel('ENPV')
# plt.title('Centralised ENPV Values - Uncertain')
# plt.legend()
# plt.show()

#Decentralised Bar Graph - Certain

# title = ['Fixed', 'Phased', 'Flexible']
# a = [488.5560, 625.7433, 694.1428] #Unopti
# b = [576.2060, 675.2027, 1017.6749] #Opti
# width = 0.6
# plt.bar(title, b, label='Optimised', width=width)
# plt.bar(title, a, label='Unoptimised', width=width)
# plt.ylabel('NPV')
# plt.title('Decentralised NPV Values - Certain')
# plt.legend()
# plt.show()

#Decentralised Bar Graph - Uncertain

# title = ['Fixed', 'Phased', 'Flexible']
# a = [642.6727, 730.1971, 798.0308] #Unopti
# b = [761.7617, 851.1134, 1135.6080] #Opti
# width = 0.6
# plt.bar(title, b, label='Optimised', width=width)
# plt.bar(title, a, label='Unoptimised', width=width)
# plt.ylabel('ENPV')
# plt.title('Decentralised ENPV Values - Uncertain')
# plt.legend()
# plt.show()







#line graphs

#Centralised Line Graph - Certain

# title = ['Fixed', 'Phased', 'Flexible']
# a = [879.4503, 957.2670, 989.3178] #Unopti
# b = [903.9705, 1049.6555, 1182.6733] #Opti
# width = 0.6
# plt.plot(title, a, color="blue", markerfacecolor="purple", linewidth=3, marker="o", markersize="10", label="Unoptimised")
# plt.plot(title, b, color="red", markerfacecolor="orange", linewidth=3, marker="o", markersize="10", label="Optimised")
# plt.ylabel('NPV')
# plt.title('Centralised NPV Values - Certain')
# plt.legend()
# plt.show()

#Centralised Line Graph - Uncertain

# title = ['Fixed', 'Phased', 'Flexible']
# a = [946.0850, 989.4617, 1051.2945] #Unopti
# b = [990.8074, 1124.6001, 1294.7958] #Opti
# plt.plot(title, a, color="blue", markerfacecolor="purple", linewidth=3, marker="o", markersize="10", label="Unoptimised")
# plt.plot(title, b, color="red", markerfacecolor="orange", linewidth=3, marker="o", markersize="10", label="Optimised")
# plt.ylabel('ENPV')
# plt.title('Centralised ENPV Values - Uncertain')
# plt.legend()
# plt.show()

#Decentralised Line Graph - Certain

# title = ['Fixed', 'Phased', 'Flexible']
# a = [488.5560, 625.7433, 694.1428] #Unopti
# b = [576.2060, 675.2027, 1017.6749] #Opti
# plt.plot(title, a, color="blue", markerfacecolor="purple", linewidth=3, marker="o", markersize="10", label="Unoptimised")
# plt.plot(title, b, color="red", markerfacecolor="orange", linewidth=3, marker="o", markersize="10", label="Optimised")
# plt.ylabel('NPV')
# plt.title('Decentralised NPV Values - Certain')
# plt.legend()
# plt.show()

#Decentralised Line Graph - Uncertain

# title = ['Fixed', 'Phased', 'Flexible']
# a = [642.6727, 730.1971, 798.0308] #Unopti
# b = [761.7617, 851.1134, 1135.6080] #Opti
# plt.plot(title, a, color="blue", markerfacecolor="purple", linewidth=3, marker="o", markersize="10", label="Unoptimised")
# plt.plot(title, b, color="red", markerfacecolor="orange", linewidth=3, marker="o", markersize="10", label="Optimised")
# plt.ylabel('ENPV')
# plt.title('Decentralised ENPV Values - Uncertain')
# plt.legend()
# plt.show()