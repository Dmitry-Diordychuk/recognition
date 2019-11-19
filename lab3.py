from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

digits = load_digits()



x_train, x_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=1/4)

logr = LogisticRegression() #C=1 solver="sag" | metrics='11'

logr.fix(x_train, y_train)

digit = x_test[0]
print(logr.predict(digit))
plt.imshow(digits.data[0].reshape(0,0), cmap='Greus_r')
plt.show()