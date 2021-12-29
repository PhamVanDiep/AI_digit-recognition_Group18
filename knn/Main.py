import glob
from tkinter import *
from tkinter import filedialog
import cv2
import numpy as np
from PIL import ImageGrab
from matplotlib import pyplot as plt
from mnist import MNIST

# 11 - 18: khởi tạo GUI
root = Tk() 
root.geometry('648x516+0+0')
root.resizable(0, 0)

root.title("Nhận diện chữ số viết tay sử dụng KNN")

width = 640
height = 480

#khởi tạo dữ liệu training

# dataset này đã được training thành mảng các số nguyên 2 chiều
mnist = MNIST()
images_train, labels_train = mnist.load_training()

# chuyển đổi dữ liệu training thành kiểu numpy.ndarray của np.float32
x_train = np.asarray(images_train).astype(np.float32)
y_train = np.asarray(labels_train).astype(np.int32)

# khởi tạo model
knn = cv2.ml.KNearest_create()
#train model với tập dữ liệu và nhãn phía trên
knn.train(x_train, cv2.ml.ROW_SAMPLE, y_train)
k = 3 #khởi tạo số lượng hàng xóm

#Tạo bảng để vẽ
cv = Canvas(root, width=width, height=height, bg='white')
cv.grid(row=0, column=0, pady=2, sticky=W, columnspan=4)

# 41 - 44: khởi tạo font
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.5
color = (255, 0, 0)
thickness = 1

def clear_widget(): #Hàm xoá toàn bộ những gì đã vẽ
    global cv
    cv.delete('all')
    print("Đã xoá")
    print("---------------------------------------------------")
# 52 - 66 để vẽ lên bảng
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

#in kết quả ra console
def put_text_result(count):
    print("Có tất cả:", count,"số được nhận diện")
    f = open('text.txt', 'r')
    data1 = f.read()
    print("Những số nhận diện được: ", data1)
    print("---------------------------------------------------")
    cv2.waitKey(0)

# Hàm nhận diện ảnh với tham số truyền vào là một số
# Số 1 là để hàm thực hiện nhận diện số khi vẽ lên bảng
# Số 2 là để hàm thực hiện nhận diện số trong ảnh nền trắng
# Số 3 là để hàm thực hiện nhận diện số trong ảnh nền đen
def Recognize_Digit(type_img):
    a = []
    # 82 - 90: lưu ảnh cần nhận diện vào biến image 
    if(type_img != 1):
        #Lấy đường dẫn của ảnh cần đọc
        fileName = filedialog.askopenfilename(initialdir = "/", title="Select A File",filetype=(("all","*.*"),("jpeg","*.jpg"),("png","*.png")))        
        for img in glob.glob(fileName): #Lấy file ảnh từ đường dẫn và lưu ảnh vào biến image
            image = cv2.imread(img, cv2.IMREAD_COLOR)    
    else: 
        filename = f'temp.png' # Lưu lại bảng với tên file: temp.png
        image = cv2.imread(filename, cv2.IMREAD_COLOR)

    if(type_img == 3): image = ~image #Nếu là ảnh nền đen thì đưa nó về ảnh nền trắng chữ đen

    gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    tmp, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # chuyển giá trị màu của ảnh về hai màu là trắng hoặc đen(giá trị 0 hoặc 255, 
    # không có giá trị xám(VD 1 2 3 125 về 0, 128 129 254 về 255))

    contours = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    # Xác định những số có trong ảnh bằng cách tìm những mã màu 255(trắng), 
    # đứng cạnh nhau và cho nó vào một khung hình chữ nhật

    f = open('text.txt', 'w+')

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if(w > 2 and h > 6):
            # 109 - 112: resize ảnh về kích thước 28x28 và đưa về mảng 1x784 kiểu float
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)
            digit = th[y:y + h, x:x + w]
            resized_digit = cv2.resize(digit, (18, 18))
            padded_digit = np.pad(resized_digit, ((5, 5), (5, 5)), "constant", constant_values=0)
            img = np.array(padded_digit)
            rec_img = img.reshape(-1,28*28).astype(np.float32)

            #Bắt đầu thực hiện tìm hàng xóm gần nhất(với k hàng xóm)
            return_value, results, neighbors, distances = knn.findNearest(rec_img, k)
            result = str(results.astype(int)[0][0]) # ép kiểu cho kết quả trả về ra số nguyên và -> kiểu string 
            print(result)       # in ra kết quả của số được nhận dạng
            print(neighbors)    # in ra các hàng xóm gần nhất với số lượng đã được định nghĩa ở trên
            print(distances)    # in ra khoảng cách tương ứng từ mục tiêu tới các hàng xóm
# 123 - 135: lưu kết quả và hiển thị kết quả ra màn hình + console
            b = {'distance': x*x + y*y, 'value': result}
            a.append(b)
            cv2.putText(image, result, (x, y - 5), font, fontScale, color, thickness)

    a.sort(key= lambda x: x.get('distance'))
    count = 0
    for i in a:
        f.write(i.get('value'))
        count += 1        
    cv2.imshow('OCR', image)
    f.close()                   
    put_text_result(count)

def OCR_on_table(): # Nhận diện số trên bảng
    # 133 - 143: crop bảng và lưu vào file temp.png
    filename = f'temp.png'
    widget = cv
    
    x = widget.winfo_rootx() + 5
    y = widget.winfo_rooty() + 5
    x1 = x + widget.winfo_width() + 155  # Dành cho máy độ phân giải 1920x1080
    y1 = y + widget.winfo_height() + 120
    # x1 = widget.winfo_width() # Dành cho máy độ phân giải 1366x768
    # y1 = widget.winfo_height()

    ImageGrab.grab().crop((x, y, x1, y1)).save(filename) #Lưu ảnh mình vẽ được lại
    Recognize_Digit(1)
    
def upload_image(): # Hàm upload với ảnh có nền trắng
    Recognize_Digit(2)

def upload_black_background_image(): # Hàm upload với ảnh có nền đen
    Recognize_Digit(3)

#4 button trong giao diện
btn_save = Button(text='Nhận dạng', command = OCR_on_table)
btn_save.grid(row=2, column=0, pady=1, padx=1)
button_clear = Button(text='Xoá', command = clear_widget)
button_clear.grid(row=2, column=1, pady=1, padx=1)
btn_save = Button(text='Tải ảnh lên', command = upload_image)
btn_save.grid(row=2, column=2, pady=1, padx=1)
btn_save = Button(text='Ảnh nền đen', command = upload_black_background_image)
btn_save.grid(row=2, column=3, pady=1, padx=1)

root.mainloop() #Hàm chính giúp giao diện được hiển thị và chương trình luôn chạy