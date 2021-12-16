from tensorflow.keras import layers
from tensorflow.keras import models
from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt
import cv2


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# print(train_labels[0])
#Load dữ liệu training and test
#train_images là mảng 60000 ảnh, train labels là gán nhãn số đó, ví dụ ảnh 1 là số 5 thì train_labels[0]=5
#tương tự với test_image là mảng 10000 ảnh
# Đây là tập dữ liệu gồm 60.000 hình ảnh thang độ xám 28x28 gồm 10 chữ số, cùng với tập hợp thử nghiệm gồm 10.000 hình ảnh
# train_images : mảng dữ liệu hình ảnh chứa dữ liệu huấn luyện. Giá trị pixel nằm trong khoảng từ 0 đến 255.
# train_labels : mảng gồm các nhãn chữ số (số nguyên trong phạm vi 0-9) 
# test_images :mảng dữ liệu hình ảnhchứa dữ liệu thử nghiệm. Giá trị pixel nằm trong khoảng từ 0 đến 255.
# test_label : Mảng gồm các nhãn chữ số (số nguyên trong phạm vi 0-9)

# Ví dụ in ra một ảnh bất kỳ trong 60000 ảnh training
# cv2.imshow('',train_images[3])
# cv2.waitKey()




#Xử lý dữ liệu đầu vào:
train_images = train_images.reshape((60000, 28, 28, 1)) 
#đặt lại kích thước dữ liệu training về 28*28 pixel, cho về ảnh đen trắng nên có độ sâu là 1 (
# Thông thước màu RGB(255,255,255), giờ chuyển về đen trắng còn (0..255))
train_images = train_images.astype('float32') / 255 
# chuyển kiểu dữ liệu về float32 và phạm vị [0..255] thành [0..1]

#Tương tự với dữ liệu test
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255


train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
# chuyển gán nhãn về mảng nhị phân 10 phần tử, ví dụ train_labels=5 => to_categorical(train_labels)= [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.] (đại diện cho {1,2,3,4,5,6,7,8,9,0})
                                                    #train_labels=2 => to_categorical(train_labels)= [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
# tức là giá trị đó bằng bao nhiêu thì a[giá trị - 1 ] = 1, còn lại bằng 0



model = models.Sequential()
#Khởi tạo một model và có thể add các lớp vào model đó



model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
# thêm  một layers vào models
# Hàm Conv2d(a,(b,c), activation='relu', input_shape=(28, 28, 1))
    # a: số filter của convolution(the number of convolution filters to use) số lượng bộ lọc chập để sử dụng
    # b,c: số hàng, cột của mỗi nhân 
    # activation='relu': hàm kích hoạt lọc các giá trị nhỏ hơn 0
    # input_shape: đầu vào, là ảnh kích thước 28*28, 1 ứng với màu đen trắng

model.add(layers.MaxPooling2D((2, 2)))
# sử dụng để làm giảm param khi train, nhưng vẫn giữ được đặc trưng của ảnh.
#ở đây là thay thế mảng 2*2 bẳng 1 phần tử đại diện

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#Như trên

model.add(layers.Flatten())
#chuyển mảng hai chiều n*n về mảng một chiều

model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
#Có 2 lớp
# tham số đầu tiên là kích thước đầu ra của lớp. Keras tự động xử lý các kết nối giữa các lớp. activation nghiên cứu sau
# lớp cuối cùng có kích thước đầu ra là 10, tương ứng với 10 lớp chữ số.

model.summary()
#show ra model vừa tạo, show các lớp của nó


model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
# biên dịch model
model.fit(train_images, train_labels, epochs=10, batch_size=64)
# Hàm fit: train_images, train_label là dữ liệu vào
#   epochs là số lần duyệt qua toàn bộ mẫu test
#   batch_size là một lần duyệt 64 ảnh
# có thể thay đổi epochs và batch_size để có model chính xác hơn

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(test_acc)
#Test thử độ chính xác của model

model.save('data-training')
#lưu model với tên folder = 'data-training4'
