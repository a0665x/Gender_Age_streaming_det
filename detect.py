import cv2
import argparse
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import PIL
import threading
global count_list
import datetime

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn,faceBoxes



def people_plot(count_list,t_now):
    x = [i for i, j in count_list]
    y = [j for i, j in count_list]
    plt.clf()
    plt.bar(x, y , color = ['blue' if i >=5 else 'green' for i in y])
    plt.xlabel('Time')
    plt.ylabel('Numbers')
    plt.ylim(0, 15)
    buffer_ = BytesIO()
    plt.savefig(buffer_, format='jpg')
    count_list.append((t_now, people_cunt))
    buffer_.seek(0)
    dataPIL = PIL.Image.open(buffer_)
    plt_img = np.asarray(dataPIL)
    buffer_.close()
    return plt_img


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--image')
    args=parser.parse_args()
    faceProto="opencv_face_detector.pbtxt"
    faceModel="opencv_face_detector_uint8.pb"
    ageProto="age_deploy.prototxt"
    ageModel="age_net.caffemodel"
    genderProto="gender_deploy.prototxt"
    genderModel="gender_net.caffemodel"

    MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
    ageList=['(0-6)', '(6-16)', '(16-25)', '(25-35)', '(35-43)', '(43-53)', '(53-65)', '(65-100)']
    genderList=['Male','Female']

    faceNet=cv2.dnn.readNet(faceModel,faceProto)
    ageNet=cv2.dnn.readNet(ageModel,ageProto)
    genderNet=cv2.dnn.readNet(genderModel,genderProto)

    video=cv2.VideoCapture(args.image if args.image else 0)
    padding=20
    people_cunt = 0

    people_cunt_temp = 0
    people_cunt = 0

    fig = plt.figure()
    fig.add_subplot(1, 1, 1)
    count_list = []
    t = threading.currentThread()
    t_now = 0
    prev_sec = datetime.datetime.now().second


    while True:
        hasFrame,frame=video.read()
        if not hasFrame:
            cv2.waitKey()
            # print('no frame')
            break

        resultImg,faceBoxes=highlightFace(faceNet,frame)
        if not faceBoxes:
            cv2.imshow("Detecting age and gender", resultImg)
            # print("No face detected")
        people_cunt = people_cunt + len(faceBoxes)
        female_cunt, male_cunt = 0, 0
        for faceBox in faceBoxes:
            face=frame[max(0,faceBox[1]-padding):
                       min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                       :min(faceBox[2]+padding, frame.shape[1]-1)]

            try:
                blob=cv2.dnn.blobFromImage(face, 1.0, (228,228), MODEL_MEAN_VALUES, swapRB=False)
            except:
                continue

            genderNet.setInput(blob)
            genderPreds=genderNet.forward()
            gender=genderList[genderPreds[0].argmax()]
            # print(f'Gender: {gender}')
            if gender =='Male':
                male_cunt = male_cunt + 1
            else :
                female_cunt = female_cunt + 1


            ageNet.setInput(blob)
            agePreds=ageNet.forward()
            age=ageList[agePreds[0].argmax()]
            # print(f'Age: {age[1:-1]} years')

            cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
        cv2.putText(resultImg, f'people_cunt:{people_cunt}', (30, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(resultImg, f'male_cunt:{male_cunt}', (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,cv2.LINE_AA)
        cv2.putText(resultImg, f'female_cunt:{female_cunt}', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,cv2.LINE_AA)
        cv2.imshow("Detecting age and gender", resultImg)

        if len(count_list) >= 40:
            count_list = count_list[-40:]
        else:
            count_list = count_list + [(0, 0)] * (40 - len(count_list))

        plt_p = people_plot(count_list, t_now)

        if datetime.datetime.now().second != prev_sec:
            count_list.append((t_now, people_cunt))
            cv2.imshow("data", plt_p)

        t_now += 1
        prev_sec = datetime.datetime.now().second
        people_cunt_temp = people_cunt
        people_cunt = 0
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

