# AI_digit-recognition_Group18
  Chương trình được viết bằng ngôn ngữ Python
  Trong file có chứa hai chương trình khác nhau để so sánh độ chính xác khi nhận diện bằng hai thuật toán là CNN và KNN.
  Để chạy chương trình chính sử dụng thuật toán CNN, ta chạy file Main.py ở thư mục gốc. Để tạo một model nhận dạng mới chúng ta chạy file create_model.py( Vì
thời gian khá lâu, để tránh mất thời gian nên nhóm đã tạo sẵn model nhận diện và lưu ở trong file rồi).
  Để chạy chương trình sử dung thuật toán KNN chúng ta chạy file Main.py trong thư mục knn (model được training khi chạy chương trình). Để chạy
chương trình đánh giá hiệu quả của thuật toán ứng với lựa chọn số láng giềng ta chạy file Accuracy.py (hiệu quả được đánh giá với k từ 1 đến 9).
  Hai chương trình có giao diện giống hết nhau, Chương trình có một khung vẽ cho người sủ dụng, khi vẽ xong, để nhận dạng chúng ta nhấn vào button nhận dạng, khi đó sẽ có
một cửa sổ mới hiển thị những số nhận biết được và độ chính xác của chúng. Ta cũng có thể xoá bảng và tiếp tục vẽ và nhận dạng tiếp
  Chúng ta cũng có thể upload ảnh cần nhận diện các số lên, chương trình có thể nhận diện được cả ảnh nền trắng và nền đen.
  Sau mỗi lần nhận diện, chương trình sẽ lưu lại một ảnh về những số mình vừa viết và một file text lưu giá trị những số nhận biết được
