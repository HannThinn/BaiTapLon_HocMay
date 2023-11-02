import numpy as np #Tạo mảng
import pandas as pd #Đọc dữ liệu đầu vào
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Đọc dữ liệu từ tập dữ liệu test_scores.csv
data = pd.read_csv("test_scores.csv")

#Vẽ biểu đồ từ cột 3 và 4 từ dữ liệu
x1 = data.values[: , 3]
y1 = data.values[: , 4]
plt.scatter(x1,y1, marker = "o")
plt.show()

new_cols = ['student_id', 'n_student', 'gender', 'pretest', 'posttest']

data = data[new_cols]

data.drop(['student_id'], axis=1, inplace=True)

print("Data_Train lần thứ 1: ")
print(data)

# Chia tập dữ liệu 70% dùng để huấn luyện, 30% dùng để kiểm tra.
dt_Train, dt_Test = train_test_split(data, test_size=0.3, shuffle=True)

# Đặt k = 4 vì số dữ liệu train chia hết cho 8
k = 4;
# Khai báo tham số cho Kfold
kf = KFold(n_splits=k, random_state=None)


def lossFunction(y_pred, y):
    difArray = []
    y_array = np.array(y)
    for i in range(0, len(y_pred)):
        dif = np.abs(y_array[i] - y_pred[i])
        difArray.append(dif)

    return np.mean(difArray)


min = 999999999999999999999999
i = 1;
# Chia tập huấn luyện thành k -1 phần, một phần còn lại dùng để kiểm tra
for (train_index, validation_index) in kf.split(dt_Train):

    X_val = dt_Train.iloc[validation_index, :3]
    y_val = dt_Train.iloc[validation_index, 3]

    # Dùng hàm Linear Regression để huấn luyện mô hình.
    lr = LinearRegression()
    X_train = dt_Train.iloc[train_index, :3]
    y_train = dt_Train.iloc[train_index, 3]
    lr.fit(X_train, y_train)

    y_pred_train = lr.predict(X_train)
    y_pred_val = lr.predict(X_val)

    sum_lossFunc = lossFunction(y_pred_train, y_train) + lossFunction(y_pred_val, y_val)
    print("sum_error lần thứ", i, ": ", sum_lossFunc)

    # Lấy ra mô hình tốt nhất
    if sum_lossFunc < min:
        min = sum_lossFunc
        regr = lr.fit(X_train, y_train)
        last = i
    i = i + 1

    y_predict = regr.predict(dt_Test.iloc[:,:3])
    y = np.array(dt_Test.iloc[:,3])
# In ra kết quả.
print("w = ", regr.coef_)
print("\nw0 = ", regr.intercept_)
print("\nKết quả tối ưu thu được nằm ở lần thử thứ ", last)

y_test = dt_Test.iloc[:, 3]
X_test = dt_Test.iloc[:, :3]
y_pred_test = regr.predict(X_test)

print("\nCoefficient %.2f" % regr.score(X_test, y_test))

print("Thực tế\t Dự đoán\t\t\t Chênh lệnh")
for i in range (0, len(y)):
    print("%.2f" % y[i], "\t",y_predict[i],"\t",abs(y[i]-y_predict[i]))