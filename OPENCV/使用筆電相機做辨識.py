import cv2

# 路徑改成opencv資料夾中haarcascade_frontalface_default.xml的絕對位置
# 可使用同資料夾中的不同XML檔做不同偵測
# 也可同時使用複數偵測
detector = cv2.CascadeClassifier('D:\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture("D:/a.avi")

text = 'Howhow'

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.rectangle(img, (x, y), (x+150, y-40), (255, 0, 0), -1)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.imshow('frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()