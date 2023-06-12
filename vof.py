import matplotlib.pyplot as plt

#Basic
cenDet= [879.4503, 957.2670, 989.3178] #Unopti
cenDetOpt= [903.9705, 1049.6555, 1182.6733] #Opti

cenUn = [946.0850, 989.4617, 1051.2945] #Unopti
cenUnOpt = [990.8074, 1124.6001, 1294.7958] #Opti

decenDet = [488.5560, 625.7433, 694.1428] #Unopti
decenDetOpt= [576.2060, 675.2027, 1017.6749] #Opti

decenUn = [642.6727, 730.1971, 798.0308] #Unopti
decenUnOpt = [761.7617, 851.1134, 1135.6080] #Opti

#Volatility changes

#Discount Rate change

def calc(args):
    a = args[1] - args[0]
    b = args[2] - args[0]
    c = args[2] - args[1]
    return a, b, c

cenDetvof = calc(cenDetOpt)
cenUnvof = calc(cenUnOpt)
decenDetvof = calc(decenDetOpt)
decenUnvof = calc(decenUnOpt)

# for i in range(len(cenDet)):
#     cenDetvof[i] = cenDetOpt[i] - cenDet[i]
#     cenUnvof[i] = cenUnOpt[i] - cenUn[i]
#     decenDetvof[i] = decenDetOpt[i] - decenDet[i]
#     decenUnvof[i] = decenUnOpt[i] - decenUn[i]

title = ['Phased - Fixed', 'Flexible - Fixed', 'Flexible - Phased']
width = 0.6

# plt.bar(title, cenDetvof, label='VoF values', width=width)
# plt.title('Centralised Deterministic VoF')

# plt.bar(title, cenUnvof, label='VoF values', width=width)
# plt.title('Centralised Uncertain VoF')

# plt.bar(title, decenDetvof, label='VoF values', width=width)
# plt.title('Decentralised Deterministic VoF')

plt.bar(title, decenUnvof, label='VoF values', width=width)
plt.title('Decentralised Uncertain VoF')

plt.ylabel('VoF')
plt.legend()
plt.show()

print(cenDetvof)
print(cenUnvof)
print(decenDetvof)
print(decenUnvof)



