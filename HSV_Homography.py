########################################################################
######					TTDE-PG  main texfile				 		####
###### authors:                                                     ####
######  * Sebastian Sierra-Alarcón s.sierra11@uniandes.edu.co       ####
######  *                                                           ####
######																####
######	This file includes main tex file to create all pdf files	####
######				MODIFICADO: Mayo de 2021        				####
########################################################################
# TTDE-PG
# Copyright (C) 2014 Bogotá, Colombia
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
########################################################################

########################################################################
# Paquetes necesarios para el funcionamiento
########################################################################
import numpy as np
import cv2

########################################################################
# Variables
########################################################################
hmn = 0
hmx = 1
smn = 0
smx = 1
vmn = 0
vmx = 1

greenLower = (46, 42, 97)
greenUpper = (96, 218, 255)

# Selección de puerto en el que se encuentra conectada la camara a acceder
cap = cv2.VideoCapture(0)
# Reducción de las ventanas para ser concatenadass y leidas de manera mas rapida
cap.set(3, 320)
cap.set(4, 240)
# Creación de imagen destino
height, width = 200, 400
im_dst = np.zeros((height,width,3),dtype=np.uint8)


########################################################################
#Función que permite crear una ventana para realizar una calibración HSV,
#variando los límites de cada canal
########################################################################
def calibracion():
    global greenLower
    global greenUpper
    global hmn
    global hmx
    global smn
    global smx
    global vmn
    global vmx
    kernel = np.ones((5, 5), np.uint8)

    def nothing(x):
        pass

    # creación de los diferentes Trackbar, 2 por cada canal
    cv2.namedWindow('Parametros')
    cv2.createTrackbar('Huemin', 'Parametros', 12, 179, nothing)
    cv2.createTrackbar('Huemax', 'Parametros', 37, 179, nothing)
    cv2.createTrackbar('Satmin', 'Parametros', 96, 255, nothing)
    cv2.createTrackbar('Satmax', 'Parametros', 255, 255, nothing)
    cv2.createTrackbar('Valmin', 'Parametros', 186, 255, nothing)
    cv2.createTrackbar('Valmax', 'Parametros', 255, 255, nothing)

    # Loop infinito
    while (1):

        #Lectura y visualización de la cámara
        _, frame = cap.read()
        cv2.imshow('Original',frame)
        (h, w) = frame.shape[:2]

        # Cambio de espacio de color a HSV para una mejor selección de colores
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hue, sat, val = cv2.split(hsv)
        hmn = cv2.getTrackbarPos('Huemin', 'Parametros')
        hmx = cv2.getTrackbarPos('Huemax', 'Parametros')
        smn = cv2.getTrackbarPos('Satmin', 'Parametros')
        smx = cv2.getTrackbarPos('Satmax', 'Parametros')
        vmn = cv2.getTrackbarPos('Valmin', 'Parametros')
        vmx = cv2.getTrackbarPos('Valmax', 'Parametros')

        # Aplicación de los limites de cada canal a cada ventana
        hthresh = cv2.inRange(np.array(hue), np.array(hmn), np.array(hmx))
        sthresh = cv2.inRange(np.array(sat), np.array(smn), np.array(smx))
        vthresh = cv2.inRange(np.array(val), np.array(vmn), np.array(vmx))

        # Ventana final
        tracking = cv2.bitwise_and(hthresh, cv2.bitwise_and(sthresh, vthresh))

        # Filtros morfológicos utilizados
        dilation = cv2.dilate(tracking, kernel, iterations=1)
        closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
        closing = cv2.GaussianBlur(closing, (5, 5), 0)

        # Concatenación de las ventanas en una única ventana
        both1 = np.concatenate([hthresh, vthresh], axis=1)
        both2 = np.concatenate([sthresh, closing], axis=1)
        both = np.concatenate([both1, both2], axis=0)

        # Código para añadir los bordes
        # borderFrame=cv2.copyMakeBorder(sthresh,10,10,10,10,cv2.BORDER_CONSTANT,value=[255,255,255])
        font = cv2.FONT_HERSHEY_TRIPLEX
        cv2.putText(both, 'Tonalidad (h)', (5, 15), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(both, (125, 10), (315, 10), (255, 255, 255), (1), cv2.LINE_AA)
        cv2.line(both, (315, 10), (315, 235), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(both, (315, 235), (6, 235), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(both, (6, 235), (6, 20), (255, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(both, 'Saturacion (s)', (5, 255), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(both, (134, 250), (315, 250), (255, 255, 255), (1), cv2.LINE_AA)
        cv2.line(both, (315, 250), (315, 475), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(both, (315, 475), (6, 475), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(both, (6, 475), (6, 260), (255, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(both, 'Brillo (v)', (325, 15), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(both, (405, 10), (634, 10), (255, 255, 255), (1), cv2.LINE_AA)
        cv2.line(both, (634, 10), (634, 235), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(both, (634, 235), (325, 235), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(both, (325, 235), (325, 20), (255, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(both, 'Imagen', (325, 255), font, 0.5, (75, 45, 152), 1, cv2.LINE_AA)
        cv2.line(both, (390, 250), (634, 250), (75, 45, 152), (1), cv2.LINE_AA)
        cv2.line(both, (634, 250), (634, 475), (75, 45, 152), 1, cv2.LINE_AA)
        cv2.line(both, (634, 475), (325, 475), (75, 45, 152), 1, cv2.LINE_AA)
        cv2.line(both, (325, 475), (325, 260), (75, 45, 152), 1, cv2.LINE_AA)

        # Ventana final
        cv2.imshow('Calibracion', both)
        key = cv2.waitKey(1) & 0xFF

        hmn = cv2.getTrackbarPos('Huemin', 'Parametros')
        hmx = cv2.getTrackbarPos('Huemax', 'Parametros')
        smn = cv2.getTrackbarPos('Satmin', 'Parametros')
        smx = cv2.getTrackbarPos('Satmax', 'Parametros')
        vmn = cv2.getTrackbarPos('Valmin', 'Parametros')
        vmx = cv2.getTrackbarPos('Valmax', 'Parametros')

        # Actualización de los límites de cada canal basados en
        # los diferentes trackbar
        greenLower = (hmn,smn, vmn)
        greenUpper = (hmx, smx, vmn)

        # Criterio de finalziación
        if key == ord('q'):
            cv2.destroyAllWindows()
            break
    cap.release()
    cv2.destroyAllWindows()



########################################################################
#Llamado de la función
########################################################################

calibracion()


























