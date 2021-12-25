# Đánh giá hiệu quả của thuật toán KNN với tập dữ liệu MNIST
# Dữ liệu training là 60000 ảnh và dữ liệu test là 10000 ảnh

from mnist import MNIST
import cv2
import numpy as np

#khởi tạo dữ liệu training và dữ liệu test
#This dataset is already split into training data and test data in the form of a 2D list of integers.
mnist = MNIST('../dataset/MNIST')
images_train, labels_train = mnist.load_training()
images_test, labels_test = mnist.load_testing()

# chuyển đổi dữ liệu training và dữ liệu test thành kiểu numpy.ndarray của np.float32
x_train = np.asarray(images_train).astype(np.float32)
y_train = np.asarray(labels_train).astype(np.int32)
x_test = np.asarray(images_test).astype(np.float32)
y_test = np.asarray(labels_test).astype(np.int32)

# khởi tạo model
knn = cv2.ml.KNearest_create()
#train model với tập dữ liệu và nhãn phía trên
knn.train(x_train, cv2.ml.ROW_SAMPLE, y_train)

# Đánh giá với k = 3 (kết quả > 97% (khoảng 97,05%))
return_value, results, neighbors, distances = knn.findNearest(x_test, 3)
correct = np.count_nonzero(results.flatten() == labels_test)
accuracy = correct * 100.0 / len(labels_test)
print (accuracy)