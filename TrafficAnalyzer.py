import cvzone
import numpy
from ultralytics import YOLO
import cv2
import math
from sort import *

#Путь к видео и маске
data_path = "Video/input_video.mp4"
mask = cv2.imread("Video/frame_mask.jpg")

cap = cv2.VideoCapture(data_path)

#Определяем модель
model = YOLO('YOLO-weights/yolov8n.pt')
classNames = model.names

# Tracker
tracker = Sort(max_age=30) #параметр "запоминания" объекта на max_age кадров

#Координаты линий-индикаторов пересечения
limits_left = [200, 450, 600, 450]
limits_right = [690, 450, 1080, 450]

#Счётчики классов для двух направдений движения
carsCount_left = []
trucksCount_left = []

carsCount_right = []
trucksCount_right = []

#Цикл обработки видеопотока
while True:
    succsess, img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask) #Задаем область кадра для распознавания

    results = model(imgRegion, stream=True)
    detections = numpy.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:

            # ClassName
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            print(conf)

            if (currentClass == "truck" or currentClass == "car") and conf > 0.3:
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # CornerRect
                w, h = x2 - x1, y2 - y1
                cvzone.cornerRect(img, (x1, y1, w, h), l=10)
                cvzone.putTextRect(img, f'{currentClass} {conf}', (x1, y1 - 20),
                scale = 0.8,
                thickness=1,
                offset=3)
                currentArray = numpy.array([x1, y1, x2, y2, conf])
                detections = numpy.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)

    #Рисуем линии-индикаторы пересечений
    cv2.line(img, (limits_left[0], limits_left[1]), (limits_left[2], limits_left[3]),
             (0, 255, 0), 5)
    cv2.line(img, (limits_right[0], limits_right[1]), (limits_right[2], limits_right[3]),
             (0, 255, 0), 5)

    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
        print(result)

        #Отмечаем точки в центре распознанных объектов
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (0, 0, 255), cv2.FILLED)

        #Считаем пересечения
        if (limits_left[0] < cx < limits_left[2] and limits_left[1] - 2 <= cy <= limits_left[3] + 3):
            if currentClass == "car":
                carsCount_left.append(id)
            if currentClass == "truck":
                trucksCount_left.append(id)

        if (limits_right[0] < cx < limits_right[2] and limits_right[1] - 2 <= cy <= limits_right[3] + 3):
            if currentClass == "car":
                carsCount_right.append(id)
            if currentClass == "truck":
                trucksCount_right.append(id)


    #Рисуем счётчики автомобилей
    cvzone.putTextRect(img, f'cars: {len(carsCount_left)}', (50, 50))
    cvzone.putTextRect(img, f'trucks: {len(trucksCount_left)}', (50, 100))

    cvzone.putTextRect(img, f'cars: {len(carsCount_right)}', (700, 50))
    cvzone.putTextRect(img, f'trucks: {len(trucksCount_right)}', (700, 100))

    cv2.imshow('Image', img)
    cv2.waitKey(1)
