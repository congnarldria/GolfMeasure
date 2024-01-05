# encoding: utf-8
'''
@author: LCS
@file: img_view.py
@time: 2020/5/22 12:12
'''
import ctypes
import time
from functools import partial

import numpy as np
from PySide2.QtCore import QObject, QThread
from PySide2.QtWidgets import QApplication, QWidget, QFileDialog, QVBoxLayout
from PySide2.QtUiTools import QUiLoader
from PySide2 import QtWidgets
from PySide2 import QtCore
from PySide2 import QtGui
import cv2
import main as mx
from datetime import datetime
import json as js
import os
from PySide2.QtGui import *
from PySide2.QtCore import *
import pygame
import threading
import queue as qu




class IMG_WIN(QWidget):
    drawing = False
    ui = None

    def __init__(self, graphicsView, showPoint):
        self.x = 0
        self.y = 0
        self.drawing = False
        super().__init__()
        self.showPoint = showPoint
        self.graphicsView = graphicsView
        self.graphicsView.setStyleSheet("padding: 0px; border: 0px;")  # 内边距和边界去除
        self.scene = QtWidgets.QGraphicsScene(self)
        self.graphicsView.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)  # 改变对齐方式

        self.graphicsView.setSceneRect(0, 0, self.graphicsView.viewport().width(),
                                       self.graphicsView.height())  # 设置图形场景大小和图形视图大小一致
        self.graphicsView.setScene(self.scene)

        self.scene.mousePressEvent = self.scene_MousePressEvent  # 接管图形场景的鼠标事件
        # self.scene.mouseReleaseEvent = self.scene_mouseReleaseEvent
        self.scene.mouseMoveEvent = self.scene_mouseMoveEvent
        self.scene.wheelEvent = self.scene_wheelEvent
        self.ratio = 1  # 缩放初始比例
        self.zoom_step = 0.5  # 缩放步长
        self.zoom_max = 32  # 缩放最大值
        self.zoom_min = 0.03125  # 缩放最小值
        self.pixmapItem = None
        self.x = 0
        self.y = 0
        self.ltx = 0
        self.lty = 0
        self.gray = []

    def addScenes(self, img):  # 绘制图形
        # self.org = img
        if self.pixmapItem != None:
            originX = self.pixmapItem.x()
            originY = self.pixmapItem.y()
        else:
            originX, originY = 0, 0  # 坐标基点

        self.scene.clear()
        if img is None:
            return
        # resource = cv2.resize(img, (320, 240), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # opencv读取的bgr格式图片转换成rgb格式
        stringBuffer = img[:]

        self.pixmap = QtGui.QPixmap(QtGui.QImage(stringBuffer, img.shape[1], img.shape[0], img.shape[1] * 3,
                                                 QtGui.QImage.Format_RGB888))
        self.pixmapItem = self.scene.addPixmap(self.pixmap)
        self.pixmapItem.setScale(self.ratio)  # 缩放
        self.pixmapItem.setPos(originX, originY)
        # QApplication.processEvents()
        # self.scene.update()
        ctypes.c_long.from_address(id(stringBuffer)).value = 1

    # def scene_mouseReleaseEvent(self, event):
    #     if event.button() == QtCore.Qt.LeftButton:  # 左键释放
    #         print("鼠标左键释放")  # 响应测试语句
    #     if event.button() == QtCore.Qt.RightButton:  # 右键释放
    #         print("鼠标右键释放")  # 响应测试语句
    def scene_MousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:  # 左键按下
            # print("鼠标左键单击")  # 响应测试语句
            # print(event.scenePos())
            self.preMousePosition = event.scenePos()  # 获取鼠标当前位置
            self.pos = event.scenePos()
            self.x = self.pos.x()
            self.y = self.pos.y()
            self.showPoint()
        # if event.button() == QtCore.Qt.RightButton:  # 右键按下
        #     print("鼠标右键单击")  # 响应测试语句

    def scene_mouseMoveEvent(self, event):
        self.pos = event.scenePos()
        print("moving")
        if event.buttons() == QtCore.Qt.LeftButton:
            # print("左键移动")  # 响应测试语句
            self.MouseMove = event.scenePos() - self.preMousePosition  # 鼠标当前位置-先前位置=单次偏移量
            self.preMousePosition = event.scenePos()  # 更新当前鼠标在窗口上的位置，下次移动用
            self.pixmapItem.setPos(self.pixmapItem.pos() + self.MouseMove)  # 更新图元位置
            self.pos = event.scenePos()
            self.x = self.pos.x()
            self.y = self.pos.y()
            self.showPoint()
    # 定义滚轮方法。当鼠标在图元范围之外，以图元中心为缩放原点；当鼠标在图元之中，以鼠标悬停位置为缩放中心
    def scene_wheelEvent(self, event):
        angle = event.delta() / 8  # 返回QPoint对象，为滚轮转过的数值，单位为1/8度
        if angle > 0:
            # print("滚轮上滚")
            self.ratio += self.zoom_step  # 缩放比例自加
            if self.ratio > self.zoom_max:
                self.ratio = self.zoom_max
            else:
                w = self.pixmap.size().width() * (self.ratio - self.zoom_step)
                h = self.pixmap.size().height() * (self.ratio - self.zoom_step)
                x1 = self.pixmapItem.pos().x()  # 图元左位置
                x2 = self.pixmapItem.pos().x() + w  # 图元右位置
                y1 = self.pixmapItem.pos().y()  # 图元上位置
                y2 = self.pixmapItem.pos().y() + h  # 图元下位置
                if event.scenePos().x() > x1 and event.scenePos().x() < x2 \
                        and event.scenePos().y() > y1 and event.scenePos().y() < y2:  # 判断鼠标悬停位置是否在图元中
                    # print('在内部')
                    self.pixmapItem.setScale(self.ratio)  # 缩放
                    a1 = event.scenePos() - self.pixmapItem.pos()  # 鼠标与图元左上角的差值
                    a2 = self.ratio / (self.ratio - self.zoom_step) - 1  # 对应比例
                    delta = a1 * a2
                    self.pixmapItem.setPos(self.pixmapItem.pos() - delta)
                    # ----------------------------分维度计算偏移量-----------------------------
                    # delta_x = a1.x()*a2
                    # delta_y = a1.y()*a2
                    # self.pixmapItem.setPos(self.pixmapItem.pos().x() - delta_x,
                    #                        self.pixmapItem.pos().y() - delta_y)  # 图元偏移
                    # -------------------------------------------------------------------------

                else:
                    # print('在外部')  # 以图元中心缩放
                    self.pixmapItem.setScale(self.ratio)  # 缩放
                    delta_x = (self.pixmap.size().width() * self.zoom_step) / 2  # 图元偏移量
                    delta_y = (self.pixmap.size().height() * self.zoom_step) / 2
                    self.pixmapItem.setPos(self.pixmapItem.pos().x() - delta_x,
                                           self.pixmapItem.pos().y() - delta_y)  # 图元偏移
        else:
            # print("滚轮下滚")
            self.ratio -= self.zoom_step
            if self.ratio < 0.5:
                self.ratio = 0.5
            else:
                w = self.pixmap.size().width() * (self.ratio + self.zoom_step)
                h = self.pixmap.size().height() * (self.ratio + self.zoom_step)
                x1 = self.pixmapItem.pos().x()
                x2 = self.pixmapItem.pos().x() + w
                y1 = self.pixmapItem.pos().y()
                y2 = self.pixmapItem.pos().y() + h
                # print(x1, x2, y1, y2)
                if x1 < event.scenePos().x() < x2 \
                        and y1 < event.scenePos().y() < y2:
                    # print('在内部')
                    self.pixmapItem.setScale(self.ratio)  # 缩放
                    a1 = event.scenePos() - self.pixmapItem.pos()  # 鼠标与图元左上角的差值
                    a2 = self.ratio / (self.ratio + self.zoom_step) - 1  # 对应比例
                    delta = a1 * a2
                    self.pixmapItem.setPos(self.pixmapItem.pos() - delta)
                    # ----------------------------分维度计算偏移量-----------------------------
                    # delta_x = a1.x()*a2
                    # delta_y = a1.y()*a2
                    # self.pixmapItem.setPos(self.pixmapItem.pos().x() - delta_x,
                    #                        self.pixmapItem.pos().y() - delta_y)  # 图元偏移
                    # -------------------------------------------------------------------------
                else:
                    # print('在外部')
                    self.pixmapItem.setScale(self.ratio)
                    delta_x = (self.pixmap.size().width() * self.zoom_step) / 2
                    delta_y = (self.pixmap.size().height() * self.zoom_step) / 2
                    self.pixmapItem.setPos(self.pixmapItem.pos().x() + delta_x, self.pixmapItem.pos().y() + delta_y)
q = qu.Queue(4)


class Displayer(QObject):
    alg = None
    sinOut = Signal(object)
    canShow = 0

    def __init__(self):
        super().__init__()

    def run(self):
        print("diplay sstart")
        try:
            while True:
                QThread.msleep(1)
                if not q.empty():
                    sourceColor = q.get()
                    if not self.alg.IsOnyImage:
                        self.alg.imageProcee(sourceColor.copy())
                        self.sinOut.emit(self.alg.resourceColor)
                    else:
                        # self.sinOut.emit(self.alg.sourceColor.copy())
                        self.sinOut.emit(sourceColor.copy())
                    print(q.qsize())
                    QApplication.processEvents()
        except:
            pass


class Worker(QObject):
    alg = None
    cam1 = None
    isLive = False

    def __init__(self):
        super().__init__()

    def run(self):
        self.alg.thread_stop_event.clear()
        print("start")
        while self.alg.isLive:
            try:
                QThread.msleep(100)
                ret, source = self.cam1.read()
                q.put(source)
                # self.alg.sourceColor = cv2.imread("./Default.jpg")
                QApplication.processEvents()
            except:
                print("Error")
        print("stop")
        self.thread_stop_event.set()


class GUI(QWidget):
    def __init__(self):
        super().__init__()
        self.rcp = "Driver"
        self.ui = QUiLoader().load('pic.ui')
        self.graphic = IMG_WIN(self.ui.graphicsView, self.showPoint)  # 实例化IMG_WIN类
        self.setWindowTitle("Golf Measure")
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.ui)
        self.setLayout(self.layout)
        self.ui.checkBox.stateChanged.connect(self.oncheckBox)
        self.resize(1550, 900)
        self.assignedAngle = 0.0
        self.ui.comboBox.addItems(["color", "gray", "canny", "threshold"])
        self.ui.comboBox.currentIndexChanged.connect(self.onComboBoxChanged)
        self.ui.cbType.addItems(
            ["Driver", "brassie", "spoon", "baffy", "7_Wood", "9_Wood", "11_Wood", "iron_1", "iron_2", "iron_3",
             "iron_4"])
        self.ui.cbType.currentIndexChanged.connect(self.select_golfType)
        self.ui.pushButton.clicked.connect(self.select_img)
        self.ui.pushButton_2.clicked.connect(self.select_vid)
        self.ui.pblive.clicked.connect(self.live_stream)
        self.ui.pbSave.clicked.connect(self.saveImage)
        self.ui.pbGo.clicked.connect(self.live_stream)
        self.ui.pbPlus.clicked.connect(partial(self.pbAssignedValueSet, 1))
        self.ui.pbMinus.clicked.connect(partial(self.pbAssignedValueSet, -1))
        self.ui.VedioSlider.valueChanged.connect(self.value_changedVedio)
        self.ui.pbMeasure.clicked.connect(self.MeasureVedio)
        self.ui.pbCalibration.clicked.connect(self.calibration)
        self.ui.cbOnlyPic.stateChanged.connect(self.oncbOnlyImage)
        self.ui.CannyLowSlider.setValue(40)
        self.ui.CannyHighSlider.setValue(150)
        self.ui.sbMaskw.setValue(40)
        self.ui.sbMaskh.setValue(400)
        self.ui.sbMaskstdx.setValue(250)
        self.ui.sbMaskstdy.setValue(190)
        self.ui.sbMaskw_2.setValue(620)
        self.ui.sbMaskh_2.setValue(345)
        self.ui.sbMaskstdx_2.setValue(700)
        self.ui.sbMaskstdy_2.setValue(280)
        self.ui.sbMaskstdy.setValue(190)
        self.ui.sbMaskrout.setValue(140)
        self.ui.sbMaskrin.setValue(70)
        self.ui.sblx.setValue(281)
        self.ui.sbly.setValue(340)
        self.ui.sbMaskw_2.valueChanged.connect(self.MaskChange)
        self.ui.sbMaskh_2.valueChanged.connect(self.MaskChange)
        self.ui.sbMaskstdx_2.valueChanged.connect(self.MaskChange)
        self.ui.sbMaskstdy_2.valueChanged.connect(self.MaskChange)
        self.ui.sbMaskstdx.valueChanged.connect(self.MaskChange)
        self.ui.sbMaskstdy.valueChanged.connect(self.MaskChange)
        self.ui.sbMaskh.valueChanged.connect(self.MaskChange)
        self.ui.sbMaskw.valueChanged.connect(self.MaskChange)
        self.ui.sbMaskrout.valueChanged.connect(self.MaskChange)
        self.ui.sbMaskrin.valueChanged.connect(self.MaskChange)
        self.ui.sbFrame.valueChanged.connect(self.FrameChange)
        self.ui.pbSavercp.clicked.connect(self.SaveRecipe)
        self.ui.sblx.valueChanged.connect(self.MaskChange)
        self.ui.sbly.valueChanged.connect(self.MaskChange)
        self.actionList = []
        self.ui.actionDriver.triggered.connect(partial(self.golftypeAction, self.ui.actionDriver))
        self.actionList.append(self.ui.actionDriver)
        self.ui.actionbrassie.triggered.connect(partial(self.golftypeAction, self.ui.actionbrassie))
        self.actionList.append(self.ui.actionbrassie)
        self.ui.actionspoon.triggered.connect(partial(self.golftypeAction, self.ui.actionspoon))
        self.actionList.append(self.ui.actionspoon)
        self.ui.actionbaffy.triggered.connect(partial(self.golftypeAction, self.ui.actionbaffy))
        self.actionList.append(self.ui.actionbaffy)
        self.ui.action7_Wood.triggered.connect(partial(self.golftypeAction, self.ui.action7_Wood))
        self.actionList.append(self.ui.action7_Wood)
        self.ui.action9_Wood.triggered.connect(partial(self.golftypeAction, self.ui.action9_Wood))
        self.actionList.append(self.ui.action9_Wood)
        self.ui.action11_Wood.triggered.connect(partial(self.golftypeAction, self.ui.action11_Wood))
        self.actionList.append(self.ui.action11_Wood)
        self.ui.action1_2.triggered.connect(partial(self.golftypeAction, self.ui.action1_2))
        self.actionList.append(self.ui.action1_2)
        self.ui.action2.triggered.connect(partial(self.golftypeAction, self.ui.action2))
        self.actionList.append(self.ui.action2)
        self.ui.action3.triggered.connect(partial(self.golftypeAction, self.ui.action3))
        self.actionList.append(self.ui.action3)
        self.ui.action4.triggered.connect(partial(self.golftypeAction, self.ui.action4))
        self.actionList.append(self.ui.action4)
        self.ui.CannyLowSlider.valueChanged.connect(self.CannHigh)
        self.ui.CannyHighSlider.valueChanged.connect(self.CannyLow)
        self.ui.dial.valueChanged.connect(self.DialValueChange)
        self.alg = mx.algorithm()
        self.ui.pbDrawTmp.clicked.connect(self.alg.saveTemplate)
        self.cam1 = cv2.VideoCapture(0)
        self.alg.sourceColor = cv2.imread("Default.jpg")
        self.graphic.addScenes(self.alg.sourceColor)
        self.vidcap = None
        self.alg.filtround = self.ui.cbfilter.isChecked()
        self.framenum = 100
        self.ui.pbGo.setIcon(QIcon("go.png"))
        self.ui.lbAssigned.setStyleSheet('background-color: pink; color : black;')
        self.ui.lbState.setStyleSheet('background-color: black; color : lightGreen;')
        self.ui.lbMatchMax.setStyleSheet('background-color: black; color : lightGreen;')
        #=============set Camera Param==================================================
        self.cam1.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)  # auto mode
        # self.cam1.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # manual mode
        # self.cam1.set(cv2.CAP_PROP_EXPOSURE ,40)
        value = self.cam1.get(cv2.CAP_PROP_EXPOSURE)
        print(value)
        self.cam1.set(6, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
        self.cam1.set(cv2.CAP_PROP_FPS, 30)
        self.cam1.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cam1.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cam1.set(cv2.CAP_PROP_EXPOSURE, 1)
        fps = self.cam1.get(5)
        print(fps)
        # =============set Camera Param==================================================
        self.img = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.stateShow)
        self.timer.start(250)
        self.last_blinked = None
        self.flashing = False
        self.select_golfType()
        self.display_thread_list = []
        self.displayer_list = []
        self.init_displayer()

    def drawTemplate(self):
        self.graphic.drawing = True

    def oncbOnlyImage(self):
        if self.ui.cbOnlyPic.isChecked():
            self.alg.IsOnyImage = True
        else:
            self.alg.IsOnyImage = False

    def pbAssignedValueSet(self, incr):
        if self.alg.assignedAngle < -2:
            self.alg.assignedAngle = -2
        if self.alg.assignedAngle > 2:
            self.alg.assignedAngle = 2
        if 2 >= self.alg.assignedAngle >= -2:
            self.alg.assignedAngle += incr
            self.ui.lbAssigned.setText(str(self.alg.assignedAngle))
            self.alg.maskAngle = self.alg.assignedAngle
            if self.alg.sourceColor is not None:
                self.alg.imageProcee(self.alg.sourceColor, self.alg.assignedAngle)
            self.graphic.addScenes(self.alg.resourceColor)

    def select_golfType(self):
        self.rcp = self.ui.cbType.currentText()
        self.alg.rcp = self.rcp
        self.alg.last_rcp["rcp"] = self.rcp
        ret = os.path.isfile("./Recipe/" + self.rcp + ".json")
        if ret:
            self.alg.loadRecipe("./Recipe/" + self.rcp)
            self.setWindowTitle(self.rcp)
            self.uiLoad()

    def golftypeAction(self, actOn):
        self.rcp = actOn.text()
        ret = os.path.isfile("./Recipe/" + self.rcp + ".json")
        for act in self.actionList:
            act.setChecked(False)
        actOn.setChecked(True)
        if ret:
            self.alg.loadRecipe("./Recipe/" + self.rcp)
            self.alg.rcp = self.rcp
            self.setWindowTitle(actOn.text())
            self.uiLoad()

    def SaveRecipe(self):
        self.alg.saveRecipe("./Recipe/" + self.rcp)

    def oncheckBox(self):
        pass

    def saveImage(self):
        ret = cv2.imwrite("./Image/" + str(datetime.now().strftime('%Y-%m-%d %H-%M-%S-%f')) + ".jpg",
                          self.alg.sourceColor)

    def init_displayer(self):
        for i in range(4):
            self.display_thread_list.append(QThread())
            self.displayer_list.append(Displayer())

        for i in range(4):
            self.displayer_list[i].alg = self.alg
            self.displayer_list[i].sinOut.connect(self.graphic.addScenes)
            self.displayer_list[i].moveToThread((self.display_thread_list[i]))
            self.display_thread_list[i].started.connect(self.displayer_list[i].run)
            self.display_thread_list[i].start()
            print(i)

    def live_stream(self):
        # while True:
        #     ret ,self.img =  self.cam1.read()
        #     cv2.namedWindow("Input")
        #     cv2.imshow("Input", self.img)
        #     QApplication.processEvents()
        if not self.alg.isLive:
            self.alg.isLive = True
            self.alg.thread_stop_event.clear()
            self.live_thread = QThread()
            self.worker = Worker()
            self.worker.alg = self.alg
            self.worker.cam1 = self.cam1
            self.worker.moveToThread(self.live_thread)
            self.live_thread.started.connect(self.worker.run)
            self.live_thread.start()
            self.ui.pblive.setText("Stop")
            self.ui.pbGo.setIcon(QIcon("stop.png"))
        else:
            self.ui.pblive.setText("Stream Live")
            self.ui.pbGo.setIcon(QIcon("wait.png"))
            self.ui.pbGo.repaint()
            self.alg.isLive = False
            self.alg.thread_stop_event.wait(1)
            self.live_thread.quit()
            self.ui.pbGo.setIcon(QIcon("go.png"))

    def stateShow(self):
        self.ui.lbMatchMax.setText(str(self.alg.maxPercent) + "% Match")
        if self.alg.state == 0:
            self.ui.lbState.setStyleSheet('background-color: black; color : lightGreen;')
            self.ui.lbState.setText(
                self.alg.stateDic[self.alg.state] + str("  ResultAngle = ") + str(self.alg.resultAngle))
        if self.alg.state == 1:
            self.ui.lbState.setStyleSheet('background-color: black; color : red;')
            self.ui.lbState.setText(
                self.alg.stateDic[self.alg.state] + str("  ResultAngle = ") + str(self.alg.resultAngle))
        if self.alg.state == 2:
            if self.flashing:
                self.flashing = False
            else:
                self.flashing = True
            if self.flashing == False:
                self.ui.lbState.setStyleSheet('background-color: black; color : black;')
                self.ui.lbState.setText(self.alg.stateDic[2])
            else:
                self.ui.lbState.setStyleSheet('background-color: black; color : yellow;')
                self.ui.lbState.setText(self.alg.stateDic[2])
        if self.alg.state == 3:
            self.ui.lbState.setStyleSheet('background-color: black; color : red;')
            self.ui.lbState.setText(self.alg.stateDic[self.alg.state])
        if self.alg.state == 4:
            self.ui.lbState.setStyleSheet('background-color: black; color : red;')
            self.ui.lbState.setText(self.alg.stateDic[self.alg.state])

    def showPoint(self):
        x = round((self.graphic.x - self.graphic.pixmapItem.x())/ self.graphic.ratio, 2)
        y = round((self.graphic.y - self.graphic.pixmapItem.y())/ self.graphic.ratio, 2)
        self.ui.lbcood.setText("(" + str(x) + "," + str(y) + ")")
    def drawRect(self, x, y):
        # ================================================================================
        self.alg.resourceColor = self.alg.sourceColor
        ox = self.graphic.preMousePosition.x()
        oy = self.graphic.preMousePosition.y()
        # cv2.rectangle()
        self.graphic.addScenes(self.alg.resourceColor)

    def uiLoad(self):
        self.alg.uiloading = True
        self.ui.sbMaskstdx.setValue(self.alg.stdx)
        self.ui.sbMaskstdy.setValue(self.alg.stdy)
        self.ui.sbMaskh.setValue(self.alg.ch)
        self.ui.sbMaskw.setValue(self.alg.cw)
        self.ui.sbMaskstdx_2.setValue(self.alg.ix)
        self.ui.sbMaskstdy_2.setValue(self.alg.iy)
        self.ui.sbMaskw_2.setValue(self.alg.iw)
        self.ui.sbMaskh_2.setValue(self.alg.ih)
        self.ui.sbMaskrin.setValue(self.alg.crin)
        self.ui.sbMaskrout.setValue(self.alg.crout)
        self.ui.sblx.setValue(self.alg.lx)
        self.ui.sbly.setValue(self.alg.ly)
        self.alg.uiloading = False
        self.MaskChange()
        self.alg.proceeImage(None, self.alg.assignedAngle)
        self.graphic.addScenes(self.alg.resourceColor)

    def MaskChange(self):
        if not self.alg.uiloading:
            self.alg.stdx = self.ui.sbMaskstdx.value()
            self.alg.stdy = self.ui.sbMaskstdy.value()
            self.alg.ch = self.ui.sbMaskh.value()
            self.alg.cw = self.ui.sbMaskw.value()
            self.alg.ix = self.ui.sbMaskstdx_2.value()
            self.alg.iy = self.ui.sbMaskstdy_2.value()
            self.alg.iw = self.ui.sbMaskw_2.value()
            self.alg.ih = self.ui.sbMaskh_2.value()
            self.alg.crin = self.ui.sbMaskrin.value()
            self.alg.crout = self.ui.sbMaskrout.value()
            self.alg.lx = self.ui.sblx.value()
            self.alg.ly = self.ui.sbly.value()
            self.alg.imageProcee(self.alg.sourceColor, self.alg.assignedAngle)
            self.graphic.addScenes(self.alg.resourceColor)

    def DialValueChange(self):
        self.alg.assignedAngle = self.ui.dial.value() / 10.0
        self.ui.lbDial.setText(str(self.ui.dial.value() / 10.0))
        self.alg.maskAngle = self.ui.dial.value() / 10.0
        if self.alg.sourceColor is not None:
            self.alg.imageProcee(self.alg.sourceColor, self.alg.assignedAngle)
        self.graphic.addScenes(self.alg.resourceColor)

    def onComboBoxChanged(self):
        pass

    def imgTansX(self):
        index = self.ui.comboBox.currentIndex()
        if self.alg.resourceColor is not None:
            if index == 0:
                self.img = self.alg.resourceColor
            elif index == 1:
                self.img = self.alg.sourceGray
            elif index == 2:
                self.img = self.alg.sourceCanny
            elif index == 3:
                self.img = self.alg.sourceTh
            else:
                print("Combox Error")
        self.graphic.addScenes(self.img)

    def CannHigh(self):
        self.alg.cannyValMax = self.ui.CannyHighSlider.value()
        self.MeasureVedio()

    def CannyLow(self):
        self.alg.cannyValMin = self.ui.CannyLowSlider.value()
        self.MeasureVedio()

    def FrameChange(self):
        index = self.ui.sbFrame.value()
        self.ui.VedioSlider.setValue(index)
        self.ui.label_4.setText(str(index) + " / " + str(self.framenum - 1))
        self.display_vedio_img(index)

    def value_changedVedio(self):
        index = self.ui.VedioSlider.value()
        self.ui.sbFrame.setValue(index)
        self.ui.label_4.setText(str(index) + " / " + str(self.framenum - 1))
        self.display_vedio_img(index)

    def cv_imread(self, file_path):
        cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
        return cv_img

    def calibration(self):
        if self.vidcap != None:
            self.alg.display_video_extract(self.vidcap, self.ui.VedioSlider.value())
            self.graphic.addScenes(self.alg.resourceColor)
            self.alg.maskAngle = 0
            self.alg.isCali = True
            self.MeasureVedio()
            self.alg.isCali = False

    def display_vedio_img(self, Index):
        if self.vidcap != None:
            self.alg.display_video_extract(self.vidcap, Index)
            self.graphic.addScenes(self.alg.resourceColor)
        if True:
            self.MeasureVedio()
            self.alg.filtround = self.ui.cbfilter.isChecked()
        elif self.ui.comboBox.currentIndex() != 0:
            self.imgTansX()

    def select_vid(self):
        filePath, _ = QFileDialog.getOpenFileName(
            self.ui,  # 父窗口对象
            "选择你要上传的图片",  # 标题
            r"E:\picture\test",  # 起始目录
            "图片类型 (*MP4)"  # 选择类型过滤项，过滤内容在括号中
        )
        if filePath != '':
            self.vidcap = cv2.VideoCapture(filePath)
            self.framenum = int(self.vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.ui.label_4.setText(str(0) + " / " + str(self.framenum - 1))
            self.ui.VedioSlider.setMaximum(self.framenum - 1)
            # self.alg.analysis_vedio(self.vidcap)

    def MeasureVedio(self):
        if self.ui.cbTracking.isChecked():
            self.alg.startTracking = True
        else:
            self.alg.startTracking = False
        if self.alg.sourceColor is not None:
            index = self.ui.VedioSlider.value()
            self.alg.analysis_vedio(self.alg.sourceColor, index)
            self.graphic.addScenes(self.alg.resourceColor)

    def select_img(self):
        filePath, _ = QFileDialog.getOpenFileName(
            self.ui,  # 父窗口对象
            "选择你要上传的图片",  # 标题
            r"E:\picture\test",  # 起始目录
            "图片类型 (*.png *.jpg *.bmp)"  # 选择类型过滤项，过滤内容在括号中
        )
        if filePath == '':
            return
        else:
            self.alg.analysis_img_file(filePath, 0)
            # img = self.cv_imread(filePath)
            self.graphic.addScenes(self.alg.resourceColor)
