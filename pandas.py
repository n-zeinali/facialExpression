import pandas as pd
import matplotlib as plt
plt.style.use('ggplot')

header = ['epoch', 'train loss', 'train Accuracy', 'test Accuracy']
df = pd.read_csv("output/out.txt", sep=" ", header = None)
df.columns = header
ax = df[['epoch','train Accuracy', 'test Accuracy']].plot(x = 'epoch')
ax.set_xlabel("Epoch",fontsize=12)
ax.set_ylabel("Accuracy",fontsize=12)
plt.savefig('ax')
print(df)