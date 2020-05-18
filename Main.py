import sys
from PyQt5.QtWidgets import QDialog, QApplication
from dialog_2 import Ui_Form
from PyQt5.QtCore import QTimer
import math
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QDialog, QApplication
from PyQt5.uic import loadUi
from shapedetector import ShapeDetector

from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import imutils
import cv2
import numpy as np
import time

bangun_datar = ""

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def HitungLuas(bangun_datar, dimA, dimB):
    luas = 0
    if (bangun_datar == "triangle"):
        luas = 0.5 * float(dimA) * float(dimB)
        print(bangun_datar, ': ', luas)
    elif (bangun_datar == "square"):
        luas = float(dimA) * float(dimA)
        print(bangun_datar, ': ', luas)
    elif (bangun_datar == "rectangle"):
        luas = float(dimA) * float(dimB)
        print(bangun_datar, ': ', luas)
    elif (bangun_datar == "circle"):
        luas = 3.14 * (0.5 * float(dimA) * 0.5 * float(dimA))
        print(bangun_datar, ': ', luas)
    elif (bangun_datar == "pentagon"):
        akar_ = 5 * (5 + 4.47) * float(dimA)
        luas = 0.25 * math.sqrt(akar_)
        print(bangun_datar, ': ', luas)

    return luas


class AppWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.show()

        self.image = None
        self.button_started = False
        self.ui.btn_start.clicked.connect(self.start_webcam)

        self.ui.slider_threshold.valueChanged.connect(self.gantiValue)
        self.ui.edt_nilai_threshold.textChanged[str].connect(self.gantiValueSlider)
        self.ui.slider_threshold_2.valueChanged.connect(self.gantiValue_2)
        self.ui.edt_nilai_threshold_2.textChanged[str].connect(self.gantiValueSlider_2)
        self.ui.slider_OR.valueChanged.connect(self.gantiValue_WOR)
        self.ui.edt_WOR.textChanged[str].connect(self.gantiValueSlider_WOR)

    @pyqtSlot()


    def gantiValueSlider_WOR(self):
        if self.ui.edt_WOR.text() == '':
            s = 0
            self.ui.slider_OR.setValue(s)
        else:
            self.ui.slider_OR.setValue(int(self.ui.edt_WOR.text()))

    def gantiValue_WOR(self):
        s = self.ui.slider_OR.value()
        self.ui.edt_WOR.setText(str(s))

    def gantiValueSlider(self):
        if self.ui.edt_nilai_threshold.text() == '':
            s = 0
            self.ui.slider_threshold.setValue(s)
        else:
            self.ui.slider_threshold.setValue(int(self.ui.edt_nilai_threshold.text()))

    def gantiValue(self):
        s = self.ui.slider_threshold.value()
        self.ui.edt_nilai_threshold.setText(str(s))

    def gantiValueSlider_2(self):
        if self.ui.edt_nilai_threshold_2.text() == '':
            s = 0
            self.ui.slider_threshold_2.setValue(s)
        else:
            self.ui.slider_threshold_2.setValue(int(self.ui.edt_nilai_threshold_2.text()))

    def gantiValue_2(self):
        s = self.ui.slider_threshold_2.value()
        self.ui.edt_nilai_threshold_2.setText(str(s))

    def start_webcam(self):

        if (self.button_started == False):
            self.capture = cv2.VideoCapture(0)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT,240)
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH,320)
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.update_frame)
            self.timer.start(5)
            self.button_started = True
            self.ui.btn_start.setText('Stop Camera')
        else:
            self.button_started = False
            self.ui.btn_start.setText('Start Camera')
            self.stop_webcam();

    def update_frame(self):

        # global bangun_datar
        ret,self.image = self.capture.read()
        self.image = cv2.flip(self.image,1)

        frame = self.image
        bangun_datar = ''
        if ret:
            buf1 = self.image
            image = frame
            resized = imutils.resize(image, width=300)
            ratio = image.shape[0] / float(resized.shape[0])

            adjust_th = 50
            if self.ui.edt_nilai_threshold.text() != '':
                adjust_th = int(self.ui.edt_nilai_threshold.text())

            adjust_th_2 = 100
            if self.ui.edt_nilai_threshold_2.text() != '':
                adjust_th_2 = int(self.ui.edt_nilai_threshold_2.text())

            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (7, 7), 0)
            edged = cv2.Canny(blurred, adjust_th, adjust_th_2)
            edged = cv2.dilate(edged, None, iterations=1)
            edged = cv2.erode(edged, None, iterations=1)
            cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            sd = ShapeDetector()

            hasil_ = frame
            for c in cnts:
                M = cv2.moments(c)
                if ((M["m10"] == 0.0) and (M["m00"] == 0.0)):
                    print("Kosong")
                else:
                    print("Berhasil")
                    cX = int(M["m10"] / M["m00"] * ratio)
                    cY = int(M["m01"] / M["m00"] * ratio)
                    shape = sd.detect(c)
                    bangun_datar = shape
                    c = c.astype("float")
                    c *= ratio
                    c = c.astype("int")
                    # cv2.drawContours(image, [c], -1, (0,255,0), 2)
                    cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            image = frame
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (7, 7), 0)

            edged = cv2.Canny(gray, adjust_th, adjust_th_2)
            edged = cv2.dilate(edged, None, iterations=1)
            edged = cv2.erode(edged, None, iterations=1)

            cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)

            if (cnts == []):
                print("Tidak Ada Objek Terdeteksi !")
                self.displayImage(buf1,1)
            else:
                (cnts, _) = contours.sort_contours(cnts)
                pixelsPerMetric = None

                for c in cnts:

                    if cv2.contourArea(c) < 100:
                        continue

                    box = cv2.minAreaRect(c)
                    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
                    box = np.array(box, dtype="int")

                    box = perspective.order_points(box)
                    cv2.drawContours(image, [box.astype("int")], -1, (0, 255, 0), 2)

                    for (x, y) in box:
                        cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)

                    (tl, tr, br, bl) = box
                    (tltrX, tltrY) = midpoint(tl, tr)
                    (blbrX, blbrY) = midpoint(bl, br)

                    (tlblX, tlblY) = midpoint(tl, bl)
                    (trbrX, trbrY) = midpoint(tr, br)

                    cv2.circle(image, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
                    cv2.circle(image, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
                    cv2.circle(image, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
                    cv2.circle(image, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

                    cv2.line(image, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
                             (255, 0, 255), 2)
                    cv2.line(image, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
                             (255, 0, 255), 2)

                    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
                    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

                    if pixelsPerMetric is None:
                        width_OR = 0.955
                        if self.ui.edt_WOR.text() != '':
                            w_inc = float(self.ui.edt_WOR.text())
                            if (w_inc != 0.0):
                                width_OR = w_inc / 2.54
                        pixelsPerMetric = dB / width_OR
                        # pixelsPerMetric = dB / args["width"]

                    dimA = dA / pixelsPerMetric * 2.54
                    dimB = dB / pixelsPerMetric * 2.54

                    luas = HitungLuas(bangun_datar, dimA, dimB)


                    if (float(dimB) != 3.0):
                        self.ui.edt_shape_detect.setText(bangun_datar)
                        self.ui.edt_area.setText("{:.1f}cm2".format(luas))

                        cv2.putText(image, "{:.1f}cm".format(dimA),
                                    (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.65, (255, 255, 255), 2)
                        cv2.putText(image, "{:.1f}cm".format(dimB),
                                    (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.65, (255, 255, 255), 2)

                        cv2.putText(image, "{:.1f}cm2".format(luas),
                                    (int(blbrX + 15), int(blbrY) + 50), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.65, (255, 255, 255), 2)

                        hasil_ = image

                    else:
                        cv2.putText(image, "{:.1f}cm".format(dimB),
                                    (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.65, (255, 255, 255), 2)
                        hasil_ = image


            self.displayImage(hasil_,1)
            # time.sleep(0.2)


    def stop_webcam(self):
        self.timer.stop()

    def displayImage(self,img,window=1):
        qformat = QImage.Format_Indexed8
        if len(img.shape)==3 :
            if img.shape[2]==4 :
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        outImage = QImage(img,img.shape[1],img.shape[0],img.strides[0],qformat)
        #BGR>>RGB
        outImage = outImage.rgbSwapped()
        if window==1:
            self.ui.label_webcam.setPixmap(QPixmap.fromImage(outImage))
            self.ui.label_webcam.setScaledContents(True)

app = QApplication(sys.argv)
w = AppWindow()
w.show()
sys.exit(app.exec_())
