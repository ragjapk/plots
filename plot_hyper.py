import matplotlib.pyplot as plt 
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
markers = ['o', 'v', 's', '^']
colors = ['red', 'blue', 'green', 'black','orange']
dash = 'dotted'
auroc =[83.36,83.22,83.02]
auprc = [89.00,	88.90,	88.75]
acc = [74.46,	74.39,	74.12]
f1 = [77.15,	77.05,	76.77]
mean = [0.2433,	0.2199,	0.2540]
emd = [0.2561,	0.2396,	0.2533]
'''
auroc =[79.76,	83.02,	85.50]
auprc = [86.05,	88.75,	90.56]
acc = [70.45,	74.12,	76.33]
f1 = [72.62,	76.77,	78.78]
mean = [0.0027,	0.2540,	1.3269]
emd = [0.0306,	0.2533,	0.5379]
'''
per = [auroc, auprc, acc, f1]
for i in range(len(auroc)):
  plt.plot(['10', '1', '0.1'], per[i], marker = markers[i], color=colors[i], linestyle=dash)

label_row = ['DNT','DRINT','DGNT']
#label_column = ['100% Data', '20% Data','10% Data','5% Data']
rows = [mpatches.Patch(color=colors[i]) for i in range(3)]
#columns = [plt.plot([], [], markers[i], markerfacecolor='w', markeredgecolor='k')[0] for i in range(4)]
plt.legend(rows, label_row, loc='best', fancybox=True, framealpha=0.5, prop={'size': 8})

plt.title('α sensitivity on CelebA dataset')
plt.xlabel('α')
plt.ylabel('Performance')
plt.savefig('hyper.png', bbox_inches="tight")
