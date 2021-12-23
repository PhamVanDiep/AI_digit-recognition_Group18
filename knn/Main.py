import glob
from tkinter import *
from tkinter import filedialog
import cv2
import numpy as np
from PIL import ImageGrab
from matplotlib import pyplot as plt

root = Tk() #Tạo cửa số chính GUI
root.geometry('648x516+0+0') #Dùng để đặt vị trí ở góc trên bên phải màn hình, nếu màn hình độ phân giải khác mà k được thì xoá dòng này đi
root.resizable(0, 0) #không cho thay đổi kích thước của GUI

root.title("Nhận diện chữ số viết tay sử dụng KNN") #Tiêu đề giao diện

#Kích thước GUI
width = 640
height = 480

#Training dữ liệu
img = cv2.imread("digits.png",0)
cells = [np.hsplit(row, 100) for row in np.vsplit(img, 50)]
x = np.array(cells) 
train = x[:,:100].reshape(-1,400).astype(np.float32)
k = np.arange(10)
train_labels = np.repeat(k,500)[:,np.newaxis]
knn = cv2.ml.KNearest_create()
knn.train(train,0,train_labels)

cv = Canvas(root, width=width, height=height, bg='white')
cv.grid(row=0, column=0, pady=2, sticky=W, columnspan=4)
#Tạo bảng để vẽ


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

def Recognize_Digit(): #Nhận diện số trên bảng
    a = []
    filename = f'temp.png' #Lưu lại bảng với tên file: temp.png
    widget = cv
    
    #Giá trị toạ độ 4 góc của bảng, ảnh chụp của bảng sẽ lưu với toạ độ 4 góc như ở dưới
    x = widget.winfo_rootx()+5
    y = widget.winfo_rooty()+5
    #x1 = x + widget.winfo_width()+155 #cộng 155 và +120 là ảnh cắt được đẹp nhất, tuỳ máy mà giá trị này có thể khác
    #y1 = y + widget.winfo_height()+120

    x1 = widget.winfo_width() #Dành cho máy t do độ phân giải thấp
    y1 = widget.winfo_height()

    print(x, y, x1, y1)
    ImageGrab.grab().crop((x, y, x1, y1)).save(filename) #Lưu ảnh mình vẽ được lại


    #Giống hàm ở trên
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
        resized_digit = cv2.resize(digit, (20, 20))
        #ImageGrab.grab().crop((x, y, w, h)).save(cntname)
        #imgrec = cv2.imread(resized_digit,0)
        x2 = np.array(resized_digit)
        print(x2)
        test2 = resized_digit.reshape(-1,400).astype(np.float32)
        result1, result2, result3, result4 = knn.findNearest(test2, 10)
        print(result1)
        print(result2)
        print(result3)
        print(result4)

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        color = (255, 0, 0)
        thickness = 1
        cv2.putText(image, str(int(result1)), (x, y - 5), font, fontScale, color, thickness)
        f.write(str(int(result1)))
        count += 1

    print("Có tất cả:", count,"số được nhận diện")
    f.close()
    f = open('text.txt', 'r')
    data1 = f.read()
    print("Những số nhận diện được:",data1)
    print("---------------------------------------------------")
    
    cv2.imshow('Ket qua phan tich', image)
    cv2.waitKey(0)
    
def upload_image(): #Hàm upload với ảnh có nền đen
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
        resized_digit = cv2.resize(digit, (20, 20))
        x2 = np.array(resized_digit)
        print(x2)
        test2 = resized_digit.reshape(-1,400).astype(np.float32)
        result1, result2, result3, result4 = knn.findNearest(test2, 10)
        print(result1)
        print(result2)
        print(result3)
        print(result4)

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        color = (255, 0, 0)
        thickness = 1
        cv2.putText(image, str(int(result1)), (x, y - 5), font, fontScale, color, thickness)
        f.write(str(int(result1)))
        count += 1

    print("Có tất cả:", count,"số được nhận diện")
    f.close()
    f = open('text.txt', 'r')
    data1 = f.read()
    print("Những số nhận diện được:",data1)
    print("---------------------------------------------------")
    
    cv2.imshow('Ket qua phan tich', image)
    cv2.waitKey(0)

def upload_image1(): #Hàm upload với ảnh có nền đen
    a = []
    fileName = filedialog.askopenfilename(initialdir = "/", title="Select A File",filetype=(("all","*.*"),("jpeg","*.jpg"),("png","*.png")))
    #Lấy đường dẫn của ảnh cần đọc

    for img in glob.glob(fileName): #Lấy file ảnh từ đường dẫn và lưu ảnh vào biến image
        image = cv2.imread(img, 0)

    image = ~image
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
        resized_digit = cv2.resize(digit, (20, 20))
        x2 = np.array(resized_digit)
        print(x2)
        test2 = resized_digit.reshape(-1,400).astype(np.float32)
        result1, result2, result3, result4 = knn.findNearest(test2, 10)
        print(result1)
        print(result2)
        print(result3)
        print(result4)

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        color = (255, 0, 0)
        thickness = 1
        cv2.putText(image, str(int(result1)), (x, y - 5), font, fontScale, color, thickness)
        f.write(str(int(result1)))
        count += 1

    print("Có tất cả:", count,"số được nhận diện")
    f.close()
    f = open('text.txt', 'r')
    data1 = f.read()
    print("Những số nhận diện được:",data1)
    print("---------------------------------------------------")
    
    cv2.imshow('Ket qua phan tich', image)
    cv2.waitKey(0)

#4 button trong giao diện
btn_save = Button(text='Nhận dạng', command=Recognize_Digit)
btn_save.grid(row=2, column=0, pady=1, padx=1)
button_clear = Button(text='Xoá', command=clear_widget)
button_clear.grid(row=2, column=1, pady=1, padx=1)
btn_save = Button(text='Tải ảnh lên', command=upload_image)
btn_save.grid(row=2, column=2, pady=1, padx=1)
btn_save = Button(text='Ảnh nền đen', command=upload_image1)
btn_save.grid(row=2, column=3, pady=1, padx=1)

root.mainloop() #Hàm chính giúp giao diện được hiển thị và chương trình luôn chạy