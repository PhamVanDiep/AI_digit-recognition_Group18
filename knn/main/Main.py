import glob
from tkinter import *
from tkinter import filedialog
import cv2
import numpy as np
from PIL import ImageGrab
from matplotlib import pyplot as plt
from mnist import MNIST

#khởi tạo GUI
root = Tk() 
root.geometry('648x516+0+0')
root.resizable(0, 0)

root.title("Nhận diện chữ số viết tay sử dụng KNN")

width = 640
height = 480

#khởi tạo dữ liệu training

#This dataset is already split into training data in the form of a 2D list of integers.
mnist = MNIST('../dataset/MNIST')
images_train, labels_train = mnist.load_training()

# chuyển đổi dữ liệu training thành kiểu numpy.ndarray của np.float32
x_train = np.asarray(images_train).astype(np.float32)
y_train = np.asarray(labels_train).astype(np.int32)

# khởi tạo model
knn = cv2.ml.KNearest_create()
#train model với tập dữ liệu và nhãn phía trên
knn.train(x_train, cv2.ml.ROW_SAMPLE, y_train)
k = 10 #khởi tạo số lượng hàng xóm

#Tạo bảng để vẽ
cv = Canvas(root, width=width, height=height, bg='white')
cv.grid(row=0, column=0, pady=2, sticky=W, columnspan=4)

#khởi tạo font
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.5
color = (255, 0, 0)
thickness = 1

def clear_widget(): #Hàm xoá toàn bộ những gì đã vẽ
    global cv
    cv.delete('all')
    print("Đã xoá")
    print("---------------------------------------------------")

lastx, lasty = None, None

def draw_lines(event):
    global lastx, lasty
    x, y = event.x, event.y
    cv.create_line((lastx, lasty, x, y), width=8, fill='black', capstyle=ROUND, smooth=TRUE, splinesteps=12)
    lastx, lasty = x, y

def activate_event(event):
    global lastx, lasty
    cv.bind('<B1-Motion>', draw_lines)
    lastx, lasty = event.x, event.y

cv.bind('<Button-1>', activate_event)

def put_text_result(image, count):
    print("Có tất cả:", count,"số được nhận diện")
    f = open('text.txt', 'r')
    data1 = f.read()
    print("Những số nhận diện được: ", data1)
    print("---------------------------------------------------")
    cv2.imshow('OCR', image)
    cv2.waitKey(0)

def Recognize_Digit(): #Nhận diện số trên bảng
    a = []
    filename = f'temp.png' #Lưu lại bảng với tên file: temp.png
    widget = cv
    
    x = widget.winfo_rootx() + 5
    y = widget.winfo_rooty() + 5
    x1 = x + widget.winfo_width() + 155 
    y1 = y + widget.winfo_height() + 120

    # x1 = widget.winfo_width() #Dành cho máy độ phân giải 1366x768
    # y1 = widget.winfo_height()

    ImageGrab.grab().crop((x, y, x1, y1)).save(filename) #Lưu ảnh mình vẽ được lại

    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    f = open('text.txt', 'w+')

    count = 0

    for cnt in contours:

        x, y, w, h = cv2.boundingRect(cnt)
        
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)
        digit = th[y:y + h, x:x + w]
        resized_digit = cv2.resize(digit, (18, 18))
        padded_digit = np.pad(resized_digit, ((5, 5), (5, 5)), "constant", constant_values=0)

        img = np.array(padded_digit)
        rec_img = img.reshape(-1,28*28).astype(np.float32)

        #Bắt đầu thực hiện tìm hàng xóm gần nhất(với k hàng xóm)
        return_value, results, neighbors, distances = knn.findNearest(rec_img, k)
        result = str(results.astype(int)[0][0])
        print(result)
        print(neighbors)

        cv2.putText(image, result, (x, y - 5), font, fontScale, color, thickness)
        f.write(result)
        count += 1

    f.close()    
    put_text_result(image, count)




def upload_image(): #Hàm upload với ảnh có nền trắng hoặc xám
    a = []
    fileName = filedialog.askopenfilename(initialdir = "/", title="Select A File",filetype=(("all","*.*"),("jpeg","*.jpg"),("png","*.png")))
    #Lấy đường dẫn của ảnh cần đọc

    for img in glob.glob(fileName): #Lấy file ảnh từ đường dẫn và lưu ảnh vào biến image
        image = cv2.imread(img, 0)

    ret, th = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    #chuyển giá trị màu của ảnh về hai màu là trắng hoặc đen(giá trị 0 hoặc 255, không có giá trị xám(VD 1 2 3 125 về 0, 128 129 254 về 255))

    contours = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    #Xác định những số có trong ảnh bằng cách tìm những mã màu 255(trắng) đứng cạnh nhau và cho nó vào một khung hình chữ nhật

    f = open('text.txt', 'w+')
    #Mở file để ghi những số nhận diện được
    count = 0

    for cnt in contours:

        x, y, w, h = cv2.boundingRect(cnt)
        
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)
        digit = th[y:y + h, x:x + w]
        resized_digit = cv2.resize(digit, (18, 18))
        padded_digit = np.pad(resized_digit, ((5, 5), (5, 5)), "constant", constant_values=0)
        x2 = np.array(padded_digit)

        img = np.array(padded_digit)
        rec_img = img.reshape(-1,28*28).astype(np.float32)

        #Bắt đầu thực hiện tìm hàng xóm gần nhất(với k hàng xóm)
        return_value, results, neighbors, distances = knn.findNearest(rec_img, k)
        result = str(results.astype(int)[0][0])
        print(result)
        print(neighbors)

        cv2.putText(image, result, (x, y - 5), font, fontScale, color, thickness)
        f.write(result)
        count += 1

    f.close()
    put_text_result(image, count)

def upload_black_background_image(): #Hàm upload với ảnh có nền đen
    #176-190 tương tự hàm bên trên
    a = []
    fileName = filedialog.askopenfilename(initialdir = "/", title="Select A File",filetype=(("all","*.*"),("jpeg","*.jpg"),("png","*.png")))
    for img in glob.glob(fileName):
        image = cv2.imread(img, 0)
    image = ~image
    ret, th = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    f = open('text.txt', 'w+')

    count = 0

    for cnt in contours:

        x, y, w, h = cv2.boundingRect(cnt)
        
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)
        digit = th[y:y + h, x:x + w]
        resized_digit = cv2.resize(digit, (18, 18))
        padded_digit = np.pad(resized_digit, ((5, 5), (5, 5)), "constant", constant_values=0)
        x2 = np.array(padded_digit)

        img = np.array(padded_digit)
        rec_img = img.reshape(-1,28*28).astype(np.float32)

        #Bắt đầu thực hiện tìm hàng xóm gần nhất(với k hàng xóm)
        return_value, results, neighbors, distances = knn.findNearest(rec_img, k)
        result = str(results.astype(int)[0][0])
        print(result)
        print(neighbors)

        cv2.putText(image, result, (x, y - 5), font, fontScale, color, thickness)
        f.write(result)
        count += 1

    f.close()
    put_text_result(image, count)

#4 button trong giao diện
btn_save = Button(text='Nhận dạng', command = Recognize_Digit)
btn_save.grid(row=2, column=0, pady=1, padx=1)
button_clear = Button(text='Xoá', command = clear_widget)
button_clear.grid(row=2, column=1, pady=1, padx=1)
btn_save = Button(text='Tải ảnh lên', command = upload_image)
btn_save.grid(row=2, column=2, pady=1, padx=1)
btn_save = Button(text='Ảnh nền đen', command = upload_black_background_image)
btn_save.grid(row=2, column=3, pady=1, padx=1)

root.mainloop() #Hàm chính giúp giao diện được hiển thị và chương trình luôn chạy