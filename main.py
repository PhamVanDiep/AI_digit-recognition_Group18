# Lưu ý khi chạy chương trình
# Nên để khung giao diện ở góc trên bên phải màn hình để ảnh cắt được chính xác nhất
# Nếu để chương trình ở chỗ khác ảnh nó sẽ cắt thiếu chỗ này thừa chỗ kia
# Tuỳ độ phân giải của máy mà toạ độ x1, y1 (dòng 168 của hàm Recognize_Digit()) sẽ cộng hoặc trừ với một giá trị nào đó để ảnh được cắt chuẩn nhất
import glob
from tkinter import *
from tkinter import filedialog
import cv2
import numpy as np
from PIL import ImageGrab
from tensorflow.keras.models import load_model

#Sẽ có một file python riêng chỉ để training phần nhận diện, vào lưu model nhận diện này vào folder data-training 
model = load_model('data-training')
#Load model vừa training
# model = load_model('data-training1')


root = Tk() #Tạo cửa số chính GUI
root.geometry('648x516+0+0') #Dùng để đặt vị trí ở góc trên bên phải màn hình, nếu màn hình độ phân giải khác mà k được thì xoá dòng này đi
root.resizable(0, 0) #không cho thay đổi kích thước của GUI

root.title("Nhận diện chữ số viết tay") #Tiêu đề giao diện

#Kích thước GUI
width = 640
height = 480



cv = Canvas(root, width=width, height=height, bg='white')
cv.grid(row=0, column=0, pady=2, sticky=W, columnspan=4)
#Tạo bảng để vẽ


def clear_widget(): #Hàm xoá toàn bộ những gì đã vẽ
    global cv
    cv.delete('all')
    print("Đã xoá")
    print("---------------------------------------------------")

lastx, lasty = None, None

#Hàm vẽ số lên bảng
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
def upload_image(): #Hàm upload với ảnh có nền trắng hoặc màu nhạt như màu trắng
    a = []
    fileName = filedialog.askopenfilename(initialdir = "/", title="Select A File",filetype=(("all","*.*"),("jpeg","*.jpg"),("png","*.png")))
    print(fileName) #đường dẫn có tiếng việt nó k đọc được
    #Lấy đường dẫn của ảnh cần đọc

    for img in glob.glob(fileName): #Lấy file ảnh từ đường dẫn và lưu ảnh vào biến image
        image = cv2.imread(img, cv2.IMREAD_COLOR)
        # print(image)
        # cv2.imshow('Display Image', image)
        # cv2.waitKey(0)
        # image = cv2.resize(image, (640, 480)) #Resize theo 2 chiều có thể làm méo ảnh, 
    gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY) 
    # cv2 giúp chuyển đổi màu ảnh về màu trắng đen

    ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    #chuyển giá trị màu của ảnh về hai màu là trắng hoặc đen(giá trị 0 hoặc 255, không có giá trị xám(VD 1 2 3 125 về 0, 128 129 254 về 255))

    contours = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    #Xác định những số có trong ảnh bằng cách tìm những mã màu 255(trắng) đứng cạnh nhau và cho nó vào một khung hình chữ nhật

    f = open('text.txt', 'w+')
    #Mở file để ghi những số nhận diện được

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if(w > 2 and h > 6): #Nếu kích thước khung hình lớn hơn 18*18px thì mới nhận diện,tránh nhận diện cái vớ vẩn

            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)
            digit = th[y:y + h, x:x + w]
            resized_digit = cv2.resize(digit, (18, 18))
            padded_digit = np.pad(resized_digit, ((5, 5), (5, 5)), "constant", constant_values=0)
            digit = padded_digit.reshape(1, 28, 28, 1)
            digit = digit / 255.0
            pred = model.predict([digit])[0]
            final_pred = np.argmax(pred)
            a2 = (int)(np.argsort(pred, axis=0)[-2])
            # data = str(final_pred) + ' ' + str(int(max(pred) * 100)) + '%'  # Muốn hiển thị cả độ chính xác thì dùng dòng này
            if(int(pred[a2] * 10000) > 100):
                data = str(final_pred) + ' ' + str(int(max(pred) * 100)) + '%, ' + str(a2) + ' ' + str(int(pred[a2] * 10000)/100) + '%'
            else: 
                data = str(final_pred) + ' ' + str(int(max(pred) * 100)) + '% '
            # data = str(final_pred) # Còn k thì dùng dòng này
            b = {'distance': x*x + y*y, 'value': str(final_pred)}
            a.append(b)
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.5
            color = (255, 0, 0)
            thickness = 1
            cv2.putText(image, data, (x, y - 5), font, fontScale, color, thickness)

    a.sort(key= lambda x: x.get('distance'))
    # print(a)
    count = 0
    for i in a:
        f.write(i.get('value'))
        count += 1
    print("Có tất cả:", count,"số được nhận diện")
    f.close()
    f = open('text.txt', 'r')
    data1 = f.read()
    print("Những số nhận diện được:",data1)
    print("---------------------------------------------------")
    cv2.imshow('image', image)
    cv2.waitKey(0)
    #Hiển thị ảnh ban đầu và dự đoán số


#Hàm này không khác gì upload_image, khác mỗi chỗ đổi lại tông màu ảnh đen về trắng
def upload_image1():
    a = []
    fileName = filedialog.askopenfilename(initialdir = "/", title="Select A File",filetype=(("all","*.*"),("jpeg","*.jpg"),("png","*.png")))
    print(fileName) #đường dẫn có tiếng việt nó k đọc được
    for img in glob.glob(fileName):
        image = cv2.imread(img, cv2.IMREAD_COLOR)
        # print(image)
        # cv2.imshow('Display Image', image)
        # cv2.waitKey(0)
        # image = cv2.resize(image, (640, 480)) #Resize theo 2 chiều có thể làm méo ảnh, 
    image = ~image
    gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    f = open('text.txt', 'w+')
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        print(w,h)
        if(w > 2 and h > 6): #Nếu kích thước khung hình lớn hơn 18*18px thì mới nhận diện,tránh nhận diện cái vớ vẩn

            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)
            digit = th[y:y + h, x:x + w]
            resized_digit = cv2.resize(digit, (18, 18))
            padded_digit = np.pad(resized_digit, ((5, 5), (5, 5)), "constant", constant_values=0)
            digit = padded_digit.reshape(1, 28, 28, 1)
            digit = digit / 255.0
            pred = model.predict([digit])[0]
            final_pred = np.argmax(pred)
            a2 = (int)(np.argsort(pred, axis=0)[-2])
            # data = str(final_pred) + ' ' + str(int(max(pred) * 100)) + '%'  # Muốn hiển thị cả độ chính xác thì dùng dòng này
            if(int(pred[a2] * 10000) > 100):
                data = str(final_pred) + ' ' + str(int(max(pred) * 100)) + '%, ' + str(a2) + ' ' + str(int(pred[a2] * 10000)/100) + '%'
            else: 
                data = str(final_pred) + ' ' + str(int(max(pred) * 100)) + '% '
            # data = str(final_pred) # Còn k thì dùng dòng này
            b = {'distance': x*x + y*y, 'value': str(final_pred)}
            a.append(b)
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.5
            color = (255, 0, 0)
            thickness = 1
            cv2.putText(image, data, (x, y - 5), font, fontScale, color, thickness)

    
    # a_orders = sorted(a.items(), key=lambda x: x[1], reverse=True)
    a.sort(key= lambda x: x.get('distance'))
    # print(a)
    count = 0
    for i in a:
        f.write(i.get('value'))
        count += 1
    print("Có tất cả:", count,"số được nhận diện")
    f.close()
    f = open('text.txt', 'r')
    data1 = f.read()
    print("Những số nhận diện được:",data1)
    print("---------------------------------------------------")
    cv2.imshow('Ket qua phan tich', image)
    cv2.waitKey(0)
    


def Recognize_Digit(): #Nhận diện số trên bảng
    a = []
    filename = f'temp.png' #Lưu lại bảng với tên file: temp.png
    widget = cv
    
    #Giá trị toạ độ 4 góc của bảng, ảnh chụp của bảng sẽ lưu với toạ độ 4 góc như ở dưới
    x = widget.winfo_rootx()+5
    y = widget.winfo_rooty()+5
    x1 = x + widget.winfo_width()+155 #cộng 155 và +120 là ảnh cắt được đẹp nhất, tuỳ máy mà giá trị này có thể khác
    y1 = y + widget.winfo_height()+120
    # x = widget.winfo_rootx()
    # y = widget.winfo_rooty()
    # x1 = x + widget.winfo_width()
    # y1 = y + widget.winfo_height()
    # print("Toạ độ root: ", root.winfo_rootx(), root.winfo_rooty())
    # print( "Toạ độ canvas: ",widget.winfo_rootx(), widget.winfo_rooty())
    # print("Kích thước canvas: ", widget.winfo_width(), root.winfo_height())

    print(x, y, x1, y1)
    ImageGrab.grab().crop((x, y, x1, y1)).save(filename) #Lưu ảnh mình vẽ được lại


    #Giống hàm ở trên
    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    f = open('text.txt', 'w+')
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
       
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)
        digit = th[y:y + h, x:x + w]
        resized_digit = cv2.resize(digit, (18, 18))
        padded_digit = np.pad(resized_digit, ((5, 5), (5, 5)), "constant", constant_values=0)
        digit = padded_digit.reshape(1, 28, 28, 1)
        digit = digit / 255.0
        pred = model.predict([digit])[0]
        final_pred = np.argmax(pred)
        a2 = (int)(np.argsort(pred, axis=0)[-2])
        # data = str(final_pred) + ' ' + str(int(max(pred) * 100)) + '%'  # Muốn hiển thị cả độ chính xác thì dùng dòng này
        if(int(pred[a2] * 10000) > 100):
            data = str(final_pred) + ' ' + str(int(max(pred) * 100)) + '%, ' + str(a2) + ' ' + str(int(pred[a2] * 10000)/100) + '%'
        else: 
            data = str(final_pred) + ' ' + str(int(max(pred) * 100)) + '% '
        # data = str(final_pred) # Còn k thì dùng dòng này
        b = {'distance': x*x + y*y, 'value': str(final_pred)}
        a.append(b)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        color = (255, 0, 0)
        thickness = 1
        cv2.putText(image, data, (x, y - 5), font, fontScale, color, thickness)
    
    # a_orders = sorted(a.items(), key=lambda x: x[1], reverse=True)
    a.sort(key= lambda x: x.get('distance'))
    # print(a)
    count = 0
    for i in a:
        f.write(i.get('value'))
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
