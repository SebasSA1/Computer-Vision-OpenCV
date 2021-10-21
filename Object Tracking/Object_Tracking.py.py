# Paquetes necesarios para el funcionamiento
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
#Variables
hmn=0
hmx=1
smn=0
smx=1
vmn=0
vmx=1

supizqx=0
supizqy=0
supderx=0
supdery=0
infizqx=0
infizqy=0
infderx=0
infdery=0

radius=0
# Metodo para la calibración del sistema:

def calibracion():
    global hmn
    global hmx
    global smn
    global smx
    global vmn
    global vmx

    kernel = np.ones((5, 5), np.uint8)
    # Lectura de la imagen de la camara
    cap = cv2.VideoCapture(1)
    #Reducción de las ventans para ser concatenadass y leidas de manera mas rapida
    cap.set(3, 320)
    cap.set(4, 240)

    def nothing(x):
        pass

    # creación de los diferentes Trackbar
    cv2.namedWindow('Parametros')
    cv2.createTrackbar('Huemin', 'Parametros', 12, 179, nothing)
    cv2.createTrackbar('Huemax', 'Parametros', 37, 179, nothing)
    cv2.createTrackbar('Satmin', 'Parametros', 96, 255, nothing)
    cv2.createTrackbar('Satmax', 'Parametros', 255, 255, nothing)
    cv2.createTrackbar('Valmin', 'Parametros', 186, 255, nothing)
    cv2.createTrackbar('Valmax', 'Parametros', 255, 255, nothing)

    while (1):
        buzz = 0
        _, frame = cap.read()
        (h, w) = frame.shape[:2]
        zeros = np.zeros((h, w), dtype="uint8")
        #Cambio de color del frame para una mejor selección de colores
        #frame[:,:,2]=0
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hue, sat, val = cv2.split(hsv)
        hmn = cv2.getTrackbarPos('Huemin', 'Parametros')
        hmx = cv2.getTrackbarPos('Huemax', 'Parametros')
        smn = cv2.getTrackbarPos('Satmin', 'Parametros')
        smx = cv2.getTrackbarPos('Satmax', 'Parametros')
        vmn = cv2.getTrackbarPos('Valmin', 'Parametros')
        vmx = cv2.getTrackbarPos('Valmax', 'Parametros')

        # Aplicación de los valores hsv a la ventana
        hthresh = cv2.inRange(np.array(hue), np.array(hmn), np.array(hmx))
        sthresh = cv2.inRange(np.array(sat), np.array(smn), np.array(smx))
        vthresh = cv2.inRange(np.array(val), np.array(vmn), np.array(vmx))
        # Ventana final
        tracking = cv2.bitwise_and(hthresh, cv2.bitwise_and(sthresh, vthresh))
        # Filtros morfologicos utilizados
        dilation = cv2.dilate(tracking, kernel, iterations=1)
        closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
        closing = cv2.GaussianBlur(closing, (5, 5), 0)

        # Detección de circulos utilizando HoughCircles sobre hsv
        circles = cv2.HoughCircles(closing, cv2.HOUGH_GRADIENT, 2, 120, param1=120, param2=50, minRadius=10, maxRadius=0)
        # Poner los circulos sobre la imagen
        if circles is not None:
            for i in circles[0, :]:
            # Detección dependiendo de la distancia y color
                if int(round(i[2])) < 30:
                    cv2.circle(frame, (int(round(i[0])), int(round(i[1]))), int(round(i[2])), (0, 255, 0), 5)
                    cv2.circle(frame, (int(round(i[0])), int(round(i[1]))), 2, (0, 255, 0), 10)
            # Rojo dependiendo de la distancia
                elif int(round(i[2])) > 35:
                    cv2.circle(frame, (int(round(i[0])), int(round(i[1]))), int(round(i[2])), (0, 0, 255), 5)
                    cv2.circle(frame, (int(round(i[0])), int(round(i[1]))), 2, (0, 0, 255), 10)
                    buzz = 1

        #intento para mostrar difetentes frames
        #Unir todas las ventanas en una sola
        both1=np.concatenate([hthresh,vthresh],axis=1)
        both2=np.concatenate([sthresh,closing],axis=1)
        both=np.concatenate([both1,both2],axis=0)
        #Codigo para hacer los bordes
        #borderFrame=cv2.copyMakeBorder(sthresh,10,10,10,10,cv2.BORDER_CONSTANT,value=[255,255,255])
        font = cv2.FONT_HERSHEY_TRIPLEX
        cv2.putText(both,'Tonalidad (h)', (5,15), font, 0.5,(255,255,255), 1, cv2.LINE_AA)
        cv2.line(both,(125,10),(315,10),(255,255,255),(1),cv2.LINE_AA)
        cv2.line(both, (315, 10), (315, 235),(255,255,255), 1, cv2.LINE_AA)
        cv2.line(both, (315, 235), (6, 235),(255,255,255), 1, cv2.LINE_AA)
        cv2.line(both, (6, 235), (6, 20),(255,255,255), 1, cv2.LINE_AA)

        cv2.putText(both,'Saturacion (s)', (5,255), font, 0.5,(255,255,255), 1, cv2.LINE_AA)
        cv2.line(both,(134,250),(315,250),(255,255,255),(1),cv2.LINE_AA)
        cv2.line(both, (315, 250), (315, 475),(255,255,255), 1, cv2.LINE_AA)
        cv2.line(both, (315, 475), (6, 475),(255,255,255), 1, cv2.LINE_AA)
        cv2.line(both, (6, 475), (6,260),(255,255,255), 1, cv2.LINE_AA)

        cv2.putText(both,'Brillo (v)', (325,15), font, 0.5,(255,255,255), 1, cv2.LINE_AA)
        cv2.line(both,(405,10),(634,10),(255,255,255),(1),cv2.LINE_AA)
        cv2.line(both, (634, 10), (634, 235),(255,255,255), 1, cv2.LINE_AA)
        cv2.line(both, (634, 235), (325, 235),(255,255,255), 1, cv2.LINE_AA)
        cv2.line(both, (325, 235), (325, 20),(255,255,255), 1, cv2.LINE_AA)

        cv2.putText(both,'Imagen', (325,255), font, 0.5,(255,255,255), 1, cv2.LINE_AA)
        cv2.line(both,(390,250),(634,250),(255,255,255),(1),cv2.LINE_AA)
        cv2.line(both, (634, 250), (634, 475),(255,255,255), 1, cv2.LINE_AA)
        cv2.line(both, (634, 475), (325, 475),(255,255,255), 1, cv2.LINE_AA)
        cv2.line(both, (325, 475), (325, 260),(255,255,255), 1, cv2.LINE_AA)

        #Precionar "q" para salir 
        cv2.imshow('Calibracion', both)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

##################################################
##################################################
#Codígo principal de funcionamiento

def automatizacion():
    x=0
    y=0
    global radius
    global supizqx
    global supizqy
    global supderx
    global supdery
    global infizqx
    global infizqy
    global infderx
    global infdery

    ancho = 600
    CoorTexto = int(ancho * 432 / 600)
    colorSeg = (0, 0, 255)
    palabra = 'Unready'
    # Lectura de la camara y seleccion de parametros
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video",
                    help="path to the (optional) video file")
    # default para poner la cola a la bola
    ap.add_argument("-b", "--buffer", type=int, default=10,
                    help="max buffer size")
    args = vars(ap.parse_args())
    greenLower = (hmn, smn, vmn)
    greenUpper = (hmx, smx, vmx)
    pts = deque(maxlen=args["buffer"])
    # src en 1 para usar una camara externa 
    if not args.get("video", False):
        vs = VideoStream(src=1).start()
    else:
        vs = cv2.VideoCapture(args["video"])
    #Original 1
    time.sleep(0.5)

    #Loop
    while True:
        frame = vs.read()
        frame = frame[1] if args.get("video", False) else frame
        if frame is None:
            break

        #Filtros
        frame = imutils.resize(frame,width=600)
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        #blurred[:,:,2]=0
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        #Mascara
        mask = cv2.inRange(hsv, greenLower, greenUpper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
    # Contornos de la bola
    # (x, y) centro
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
        #print(len(cnts))
        cnts = imutils.grab_contours(cnts)
        center = None

    # Solo procedemos si hay contornos
        if len(cnts) > 0:
        #Encontramos el contorno mas grande que pse percibe
        # y le calculamos el centro
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # Tamaño minimo del contorno para evitar ruido
            if radius > 5:
            # Grafica del contorno, aproximado con un circulo
                cv2.circle(frame, (int(x), int(y)), int(radius),
                           (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
            print("Centro:",x,y)


            font = cv2.FONT_HERSHEY_TRIPLEX
            colorSeg=(47,255,49)
            palabra='Ready'
        pts.appendleft(center)
        font = cv2.FONT_HERSHEY_TRIPLEX
        cv2.putText(frame, palabra, (5,CoorTexto), font, 1,colorSeg, 1, cv2.LINE_AA)
        cv2.line(frame,(150,430),(590,430),colorSeg,1,cv2.LINE_AA)
        cv2.line(frame, (590, 430), (590, 10),colorSeg, 1, cv2.LINE_AA)
        cv2.line(frame, (590, 10), (10, 10), colorSeg, 1, cv2.LINE_AA)
        cv2.line(frame, (10, 10), (10, 400), colorSeg, 1, cv2.LINE_AA)

        # Instruciones
        cv2.putText(frame,'INSTRUCCIONES',(20,20),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.5,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(frame,'1) Presione "q" para pasar a la ventana de calibracion ',(20,35),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.5,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(frame,'2) Calibre los diferentes parametros H S V hasta que',(20,50),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.5,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(frame,'solo se pueda ver la pelota en la ventana imagen y luego presione "q"',(35,65),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.5,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(frame,'Para cerrar el sistema presionar "q"',(20,175),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.5,(255,255,255),1,cv2.LINE_AA)
        cv2.imshow("Frame", frame)

        key = cv2.waitKey(1) & 0xFF

    # Parar el loop con 'q'
        if key == ord("q"):
            break
    if not args.get("video", False):
        vs.stop()
    #release the camera
    else:
        vs.release()
# close all windows
    cv2.destroyAllWindows()




automatizacion()
calibracion()
automatizacion()

