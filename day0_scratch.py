''' assignment for Thursday:
use height and weight as independent variables to predict gender
(inluding logistic regression function for genders)
'''
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression

df = pd.read_csv('lessons/shared-resources/heights_weights_genders.csv')
# df.plot.scatter(x='Height', y='Weight')
# plt.show()

# gender masks, male=1
df = df.sort_values('Height')
female = df['Gender'] == 'Female'
male = df['Gender'] == 'Male'
df.Gender = male.astype(int)  # set male to 1, female to 0
X = df.Height.values.reshape(-1, 1)

X_m = df[male].Height.values.reshape(-1, 1)  # len of array x 1 (vs. list)
y_m = df[male].Weight
df['predicted_weight_as_f'] = 0
df['predicted_weight_as_m'] = 0

lr_m = LinearRegression()
lr_m.fit(X_m, y_m)
df.predicted_weight_as_m = lr_m.predict(X)


X_f = df[female].Height.values.reshape(-1, 1)  # len of array x 1 (vs. list)
y_f = df[female].Weight

lr_f = LinearRegression()
lr_f.fit(X_f, y_f)
df.predicted_weight_as_f = lr_f.predict(X)

# plot female / male
ax = df[female].plot(kind='scatter', x='Height', y='Weight', c='red',
                     alpha=.2)
ax = df[male].plot(ax=ax, kind='scatter', x='Height', y='Weight', c='blue',
                   alpha=.1)

# plot lines of best fit
df.plot(ax=ax, kind='line', x='Height', y='predicted_weight_as_f', c='gray')
df.plot(ax=ax, kind='line', x='Height', y='predicted_weight_as_m', c='k')

# logistic regression: predict genders

df = df.sort_values('Height')
logistic = LogisticRegression()

# use height and weight
X = pd.np.zeros((10000, 2))
X[:, 0] = df.Height.values
X[:, 1] = df.Weight.values

x = df[['Height', 'Weight']].values.reshape(-1, 2)

print(X.shape)
print(X[5000])

# use weight only
# X = (df.Height.values * df.Weight.values).reshape(-1, 1)

logistic = logistic.fit(X, df.Gender)

df['logreg_predicted_gender'] = logistic.predict(X)
df['prob'] = logistic.predict_proba(X)[:, 1]

'''
ax = df.plot.scatter(x='Height', y='prob', s=1, c='k')
df.plot.scatter(ax=ax, x='Height', y='logreg_predicted_gender')
'''

print('rms error:')
print(((df.prob - df.Gender) ** 2).mean() ** .5)
print('accuracy:')
print(logistic.score(X, df.Gender))

predicted_male = df.logreg_predicted_gender.astype(bool)

ax = df[predicted_male].plot.scatter(ax=ax, x='Height', y='Weight',
                                     s=.5, c='black', alpha=.5)
ax = df[~predicted_male].plot.scatter(ax=ax, x='Height', y='Weight',
                                      s=.5, c='gray', alpha=.5)


print('\n ### multivariate regression: predict weight ### \n')


def pred_weight_dumb(height, gender):
    if gender == 'Male':
        return lr_m.predict(height)
    else:
        return lr_f.predict(height)

predict_weight = LinearRegression().fit(df[['Height', 'Gender']]
                                        .values.reshape(-1, 2),
                                        df.Weight)

print(pred_weight_dumb(70, 'Male'))
print(predict_weight.predict(pd.np.array([70, 1]).reshape(1, -1)))
print(pred_weight_dumb(70, 'Female'))
print(predict_weight.predict(pd.np.array([70, 0]).reshape(1, -1)))


df['e_mlvr'] = predict_weight.predict(df[['Height', 'Gender']]
                                      .values.reshape(-1, 2)) - df.Weight
e_rms = ((df.e_mlvr ** 2).mean()) ** .5
print('rms error: {}'.format(e_rms))

df.hist('e_mlvr', bins=20)

plt.show()
