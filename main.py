#Khai báo các thư viện cần thiết

from numpy import load
from os import listdir
from os.path import isdir
import cv2
from keras_facenet import FaceNet
# Khai báo Model nhận diện khuôn mặt từ thư viện keras_facenet
# Model FaceNet() của thư viện keras_facenet đã bao gồm MTCNN detection
model = FaceNet()
# Từ khuôn mặt đầu vào, tìm kiếm, so sánh 
def find_face(face):
    # Lấy embedding từ khuôn mặt đầu vào
    # Khuôn mặt đầu vào đã được phân tích các đặc điểm gồm vị trí khuôn mặt, kích thước khuôn mặt,
    # vị trí của mắt, mũi, miệng, 128 vector embedding từ hàm facenet
    
    # Lấy ra embedding của khuôn mặt đầu vào
    emb=face['embedding']
    
    # Địa chỉ tệp dữ liệu khuôn mặt đã được huấn luyện
    directory = 'embedded'
    
    # 
    check=1
    for subdir in listdir(directory):
        # path là địa chỉ truy cầm đến folder chứa embedding khuôn mặt 
        # sau khi training của mỗi người
        path = directory +'/'+ subdir + '/'
        # bỏ qua các file không phải folder
        if not isdir(path):
            continue
        # Hàm for chạy lần lượt 
        # các embedding 
        # có trong thư mục embedding của từng người
        for i in range(0,len(listdir(path))):
            # lấy ra embedding trong folder "path"
            data=load('embedded/'+subdir+'/embedding_'+ str(i) +'.npz')
            # lấy dữ liệu từ file trên, với dòng đầu tiên là embedding đầu vào
            a=data['arr_0'] 
            # Tính toán độ lệch giữa vector embedding emb của ảnh đầu vào 
            # và vector embedding đã được training
            # Nếu độ lệch <0.1 thì đưa ra tên người được nhận diện ra
            # Với độ chính xác >90%
            # Nếu độ lệch < check thì lấy tên của người đó(Tên folder chứa embedding a)
            # đồng thời gán check = độ lệch chuẩn đó 
            if (model.compute_distance(emb,a)<0.1):
                return subdir,model.compute_distance(emb,a)
            else:
                if (model.compute_distance(emb,a)<check):
                    check=model.compute_distance(emb,a)
                    name=subdir
    # Nếu độ lệch giữa các vector embedding của 2 người <0.2 thì coi là một người
    # Nếu độ lệch nhỏ nhất giữa các vector embedding của 2 người >0.2 thì kết luận người được nhận diện là người lạ(Stranger)   
    if(check>0.21):
        print (check)
        return 'Stranger',check
    else:
        print (check)
        return name,check
       # Khởi tạo đối tượng camera
cap = cv2.VideoCapture(0) 
def take_photo():


    # Vòng lặp chính
    while True:
        # Lấy ảnh từ camera
        ret, frame = cap.read()
        faces = model.extract(frame)  #chuỗi vị trí các khuôn mặt
        if (len(faces) != 0):
        #    print("no face found")
        #    return
        
            need = faces[0]
            m=0
            #Tìm max chiều rộng để lấy ra khuôn mặt lớn nhất
            for face in faces:
                x= face['box']
                if(x[2]>m):
                    m=x[2]
                    need=face
            #Lấy vị trí khuôn mặt lớn nhất
            x1,x2,y1,y2 = need['box']
            #Vẽ boudingbox khuôn mặt được chọn
            cv2.rectangle(frame, (x1, x2), (x1+y1, x2+y2), (0, 255, 0), 2)    
        # Hiển thị ảnh
        cv2.imshow("Máy ảnh", frame)

        # Xử lý sự kiện
        key = cv2.waitKey(1)

        # Thoát khi nhấn nút 'q'
        if key == ord('q'):
            break

    # Đóng camera
    cap.release()
    cv2.destroyAllWindows()
    return frame



def Main(frame_with_face):
    #Chọn khuôn mặt có kích thước lớn nhất bằng kích thước ảnh khuôn mặt
    
    #Extract vị trí các khuôn mặt từ ảnh đầu vào đã chụp
    faces = model.extract(frame_with_face)          #chuỗi vị trí các khuôn mặt
    if (len(faces) == 0):
        print("no face found")
        return
    else: 
        need = faces[0]
    m=0
    #Tìm max chiều rộng để lấy ra khuôn mặt lớn nhất
    for face in faces:
        x= face['box']
        if(x[2]>m):
            m=x[2]
            need=face
    #Lấy vị trí khuôn mặt lớn nhất
    x1,x2,y1,y2 = need['box']
    #Vẽ boudingbox khuôn mặt được chọn
    cv2.rectangle(frame_with_face, (x1, x2), (x1+y1, x2+y2), (0, 255, 0), 2)
    name,check = find_face(need)
    #Viết tên người nhận diện được
    cv2.putText(frame_with_face, name, (x1,x2),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    while True:
        # Lấy ảnh từ camera

        # Hiển thị ảnh
        cv2.imshow("Định danh", frame_with_face)

        # Xử lý sự kiện
        key = cv2.waitKey(1)

        # Thoát khi nhấn nút 'q'
        if key == ord('q'):
            break
    print(faces)
    cap.release()
    cv2.destroyAllWindows()

frame_with_face = take_photo()

Main(frame_with_face)