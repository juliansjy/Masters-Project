#How much more flexibility can a centralised flexible and decentralised flexible hydrogen production system design gain via optimisation using VoF as a metric?
import matplotlib.pyplot as plt

from cfixed import *
from cphased import *
from cflex import *
from dfixed import *
from dphased import *
from dflex import *

a = [990.8074, 1124.6001, 1294.7958, 761.7617, 851.1134, 1135.6080]         #Optimised ENPVs
b = [-172.6361, -13.2525, 163.5321, -153.2875, -16.3866, 130.0488]          #Value at Risk (10%)
c = [1989.2126, 1874.5271, 1968.4829, 1497.0938, 1504.9549, 1549.9349]      #Value at Gain (90%)

#Main CDF curves
plt.plot(bounds1, cdf1, label="Centralised Fixed", color="purple")
# plt.plot(bounds2, cdf2, label="Centralised Phased", color="red")
plt.plot(bounds3, cdf3, label="Centralised Flexible", color="blue")
plt.plot(bounds4, cdf4, label="Decentralised Fixed", color="red")
# plt.plot(bounds5, cdf5, label="Decentralised Phased", color="brown")
plt.plot(bounds6, cdf6, label="Decentralised Flexible", color="orange")

#Unoptimised ENPV lines
# plt.vlines(x = enpv1, ymin=0, ymax=max(cdf1), label="Centralised Fixed ENPV", colors="blue", linestyle="dashed")
# plt.vlines(x = enpv2, ymin=0, ymax=max(cdf2), label="Centralised Phased ENPV", colors="red", linestyle="dashed")
# plt.vlines(x = enpv3, ymin=0, ymax=max(cdf3), label="Centralised Flexible ENPV", colors="blue", linestyle="dotted")
# plt.vlines(x = enpv4, ymin=0, ymax=max(cdf4), label="Decentralised Fixed ENPV", colors="green", linestyle="dashed")
# plt.vlines(x = enpv5, ymin=0, ymax=max(cdf5), label="Decentralised Phased ENPV", colors="brown", linestyle="dashed")
# plt.vlines(x = enpv6, ymin=0, ymax=max(cdf6), label="Decentralised Flexible ENPV", colors="orange", linestyle="dotted")

#Optimised ENPV lines
plt.vlines(x = a[0], ymin=0, ymax=max(cdf1), label="Optimised Centralised Fixed ENPV", colors="purple", linestyle="dashed")
# plt.vlines(x = a[1], ymin=0, ymax=max(cdf2), label="Optimised Centralised Phased ENPV", colors="red", linestyle="dotted")
plt.vlines(x = a[2], ymin=0, ymax=max(cdf3), label="Optimised Centralised Flexible ENPV", colors="blue", linestyle="dashed")
plt.vlines(x = a[3], ymin=0, ymax=max(cdf4), label="Optimised Decentralised Fixed ENPV", colors="red", linestyle="dashed")
# plt.vlines(x = a[4], ymin=0, ymax=max(cdf5), label="Optimised Decentralised Phased ENPV", colors="brown", linestyle="dotted")
plt.vlines(x = a[5], ymin=0, ymax=max(cdf6), label="Optimised Decentralised Flexible ENPV", colors="orange", linestyle="dashed")

#Value at Risk lines
# plt.vlines(x = b[0], ymin=0, ymax=max(cdf1), label="VaR (10%) Centralised Fixed", colors="blue", linestyle="dashed")
# plt.vlines(x = b[1], ymin=0, ymax=max(cdf2), label="VaR (10%) Centralised Phased", colors="red", linestyle="dashed")
# plt.vlines(x = b[2], ymin=0, ymax=max(cdf3), label="VaR (10%) Centralised Flexible", colors="black", linestyle="dashed")
# plt.vlines(x = b[3], ymin=0, ymax=max(cdf4), label="VaR (10%) Decentralised Fixed", colors="green", linestyle="dashed")
# plt.vlines(x = b[4], ymin=0, ymax=max(cdf5), label="VaR (10%) Decentralised Phased", colors="brown", linestyle="dashed")
# plt.vlines(x = b[5], ymin=0, ymax=max(cdf6), label="VaR (10%) Decentralised Flexible", colors="magenta", linestyle="dashed")

# #Value at Gain lines
# plt.vlines(x = c[0], ymin=0, ymax=max(cdf1), label="VaG (90%) Centralised Fixed", colors="blue", linestyle="dashed")
# plt.vlines(x = c[1], ymin=0, ymax=max(cdf2), label="VaG (90%) Centralised Phased", colors="red", linestyle="dashed")
# plt.vlines(x = c[2], ymin=0, ymax=max(cdf3), label="VaG (90%) Centralised Flexible", colors="black", linestyle="dashed")
# plt.vlines(x = c[3], ymin=0, ymax=max(cdf4), label="VaG (90%) Decentralised Fixed", colors="green", linestyle="dashed")
# plt.vlines(x = c[4], ymin=0, ymax=max(cdf5), label="VaG (90%) Decentralised Phased", colors="brown", linestyle="dashed")
# plt.vlines(x = c[5], ymin=0, ymax=max(cdf6), label="VaG (90%) Decentralised Flexible", colors="magenta", linestyle="dashed")

plt.ylabel("Cumulative Distribution Function")
plt.xlabel("NPV")
plt.title("Improvements in Value of Flexibility from Optimisation")
plt.legend(fontsize="small")
plt.show()
