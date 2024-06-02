#Khai báo các thư viện cần thiết
from keras_facenet import FaceNet
import os
from os import listdir
from os.path import isdir
from numpy import savez_compressed
import pathlib
import shutil
# Khai báo Model nhận diện khuôn mặt từ thư viện keras_facenet
# Model FaceNet() của thư viện keras_facenet đã bao gồm MTCNN detection
model = FaceNet()

def loadDataSet(directory):
    # liệt kê các thư mục lưu tr
    X, y = list(), list()

    # Hàm for chạy một lượt các folder 
    # chứa tên các hình ảnh chứa khuôn mặt của cùng một người
    for subdir in listdir(directory):
        # path là địa chỉ truy cầm đến folder chứa ảnh của mỗi người
        path = directory +'/'+ subdir + '/'
        # bỏ qua các file không phải folder
        if not isdir(path):
            continue
        embeddings=list()
        # Hàm for chạy lần lượt các ảnh có trong thư mục của từng người
        for name in listdir(path):
            # lấy ra khuôn mặt trong mỗi ảnh
            faces = model.extract(path+name)
            if (len(faces)== 0):
                
                print(name + " no face found")
                continue
            else: 
                face = faces[0]
            m=0
            for each in faces:
                x= each['box']
                if(x[2]>m):
                    m=x[2]
                    face=each
            
            
            # lấy ra vector embedding của khuôn mặt trong ảnh
            embedding = face['embedding']
            # 
            embeddings.append(embedding)
            n="embedded/"+ subdir+"/"
            try:
                # Kiểm tra xem folder chứa embedding của người đang xét đã tồn tại chưa,
                # Nếu đã tồn tại thì xóa đi để cập nhật giữ liệu mới
                if pathlib.Path(path).is_dir():
                    shutil.rmtree(n)
                # Tạo folder chứa embedding của người đang xét
                os.mkdir( n)
            except Exception as e:
                # Tạo folder chứa embedding của người đang xét
                os.mkdir( n)

        # Nén các vector embedding vào các file trực thuộc các folder của người đó
        # Mỗi khuôn mặt được đặt vào một file riêng, đặt trong folder có tên là tên của người đó
        # Các folder trên đặt trong folder embedding chứa các folder embedding của mỗi người
        for each in range(0,len(embeddings)):
            # Một model của numpy nén dữ liệu thành file dưới dạng đuôi npz
            savez_compressed('embedded/'+ subdir + '/' +'embedding_'+ str(each) +'.npz',embeddings[each])
        print("cập nhật thành công")

# Gọi hàm train data
loadDataSet("data")
