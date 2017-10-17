# another dimension
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rows = []
rows.append(['dimension', 'o'] +
            ['dist{}'.format(i) for i in range(10)])

for dim in range(1, 50):
    o = np.sqrt(dim)
    row = [dim, o]
    for i in range(10):
            v1 = np.random.randn(dim)
            v2 = np.random.randn(dim)
            dist = np.sqrt(np.dot(v1 - v2, v1 - v2))
            row.append(dist)
    rows.append(row)
    print(dist)

df = pd.DataFrame.from_records(rows[1:], columns=rows[0])
df['median'] = df.median(axis=1)

ax = df.plot.scatter(x='dimension', y='median')
df.plot.line(x='dimension', y='o', ax=ax)
plt.show()
