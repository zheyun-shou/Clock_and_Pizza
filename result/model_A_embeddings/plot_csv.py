import pandas as pd

# read the csv file
filename = 'result/circ.csv'
df = pd.read_csv(filename)

# # Clean the data
# # drop all the columns with its name contains __MIN and __MAX
# df = df.loc[:, ~df.columns.str.contains('__MIN|__MAX')]
# # the data is sampled every 50 Step, so we can drop other rows
# df = df[df['Step'] % 50 == 0]
# # save the cleaned data
# df.to_csv(filename, index=False)

# plot the data using sns.lineplot
import seaborn as sns
import matplotlib.pyplot as plt
df = df.loc[:, ~df.columns.str.contains('A')]
df_melted = df.melt(id_vars='Step', var_name='Variable', value_name='Value')

# Plot the data using Seaborn
fig, ax = plt.subplots(dpi=300)
sns.lineplot(data=df_melted, x='Step', y='Value', hue='Variable', ax=ax, dashes=False)



plt.savefig('result/circ.png')
# save the plot

