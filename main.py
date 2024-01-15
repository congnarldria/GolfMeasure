# 2023/11/3  comment
# reference from GitHub https://github.com/cozheyuanzhangde/Invariant-TemplateMatching/blob/main/InvariantTM.py
# to promote stability , change method it rotates source img, not template img
# see thr calling example at the last few lines
import math
import os.path
import img_view as imgviwer
import cv2
import numpy as np
import pygame
import json as js
import os
import threading

# calling example
My_ui = None
shared_resource_lock = threading.Lock()
shared_resource_with_lock = 0
if __name__ == '__main__':
    app = imgviwer.QApplication([])
    My_ui = imgviwer.GUI()
    My_ui.show()
    app.exec_()

class algorithm():
    rcp = None
    uiloading = False
    IsOnyImage = False
    is_img_Debug = False
    sourceColor = None
    sourceGray = None
    sourceTh = None
    sourceCanny = None
    resourceColor = None
    vidcap = None
    cannyValMax = 150
    cannyValMin = 40
    thresholdValue = 60
    maskAngle = 0
    stdx = 250
    stdy = 190
    cx = 250
    cy = 190
    cw = 40
    ch = 400
    ix = 700
    iy = 280
    iw = 620
    ih = 345
    lx = 0
    ly = 0
    crin = 200
    crout = 300
    assignedAngle = 0
    CaliAngle = 0
    isLive = False
    isCali = False
    startTracking = False
    filtround = False
    isSync = True
    isLive = False
    maxPercent = 0
    rcp_path = "./Recipe/"
    last_rcp = {"last_rcp": "Drive"}
    state = 3
    stateDic = {0 : "OK", 1: "NG", 2: "Over Range", 3: "Too Fast", 4: "Bad Image"}
    dict = {"cannyValMax": 150, "cannyValMin": 40, "thresholdValue": 60, "maskAngle": 0, "stdx": 250,
            "stdy": 190,
            "cx": 250,
            "cy": 190,
            "cw": 40,
            "ch": 400,
            "ix": 700,
            "iy": 280,
            "iw": 620,
            "ih": 345,
            "lx": 281,
            "ly": 340,
            "crin": 70,
            "crout": 140,
            "CaliAngle": 0}
    tmp = None
    thread_stop_event = threading.Event()
    def __init__(self):
        self.thread_stop_event.clear()
        self.maxPercent = -1
        ret = os.path.exists(self.rcp_path)
        if not ret:
            os.mkdir(self.rcp_path)
        ret = os.path.isfile(self.rcp_path + "last_rcp.json")
        if ret:
            f = open(self.rcp_path + "last_rcp.json")
            data = js.load(f)
        else:
            self.cannyValMax = 150
            self.cannyValMin = 40
            self.thresholdValue = 60
            self.maskAngle = 0
            self.stdx = 250
            self.stdy = 190
            self.cx = 250
            self.cy = 190
            self.cw = 40
            self.ch = 400
            self.ix = 700
            self.iy = 280
            self.iw = 620
            self.ih = 345
            self.lx = 281
            self.ly = 340
            self.crin = 70
            self.crout = 140
            self.CaliAngle = 0
        self.cannyValMax = self.dict["cannyValMax"]
        self.cannyValMin = self.dict["cannyValMin"]
        self.thresholdValue = self.dict["thresholdValue"]
        self.maskAngle = self.dict["maskAngle"]
        self.stdx = self.dict["stdx"]
        self.stdy = self.dict["stdy"]
        self.cx = self.dict["cx"]
        self.cy = self.dict["cy"]
        self.cw = self.dict["cw"]
        self.ch = self.dict["ch"]
        self.ix = self.dict["ix"]
        self.iy = self.dict["iy"]
        self.iw = self.dict["iw"]
        self.ih = self.dict["ih"]
        self.lx = self.dict["lx"]
        self.ly = self.dict["ly"]
        self.crin = self.dict["crin"]
        self.crout = self.dict["crout"]
        self.CaliAngle = self.dict["CaliAngle"]
        # self.loadRecipe(self, "./Recipe/Driver.json")
        
    def loadRecipe(self, rcp="./Recipe/mydata"):

        f = open(rcp + '.json')
        self.dict = js.load(f)
        self.cannyValMax = self.dict["cannyValMax"]
        self.cannyValMin = self.dict["cannyValMin"]
        self.thresholdValue = self.dict["thresholdValue"]
        self.maskAngle = self.dict["maskAngle"]
        self.stdx = self.dict["stdx"]
        self.stdy = self.dict["stdy"]
        self.cx = self.dict["cx"]
        self.cy = self.dict["cy"]
        self.cw = self.dict["cw"]
        self.ch = self.dict["ch"]
        self.ix = self.dict["ix"]
        self.iy = self.dict["iy"]
        self.iw = self.dict["iw"]
        self.ih = self.dict["ih"]
        self.lx = self.dict["lx"]
        self.ly = self.dict["ly"]
        self.crin = self.dict["crin"]
        self.crout = self.dict["crout"]
        self.CaliAngle = self.dict["CaliAngle"]
        self.tmp = cv2.imread("./Recipe/" + self.rcp + ".jpg")
    def saveRecipe(self, rcp="./Recipe/mydata"):
        self.dict["cannyValMax"] = self.cannyValMax
        self.dict["cannyValMin"] = self.cannyValMin
        self.dict["thresholdValue"] = self.thresholdValue
        self.dict["maskAngle"] = self.maskAngle
        self.dict["stdx"] = self.stdx
        self.dict["stdy"] = self.stdy
        self.dict["cx"] = self.cx
        self.dict["cy"] = self.cy
        self.dict["cw"] = self.cw
        self.dict["ch"] = self.ch
        self.dict["ix"] = self.ix
        self.dict["iy"] = self.iy
        self.dict["iw"] = self.iw
        self.dict["ih"] = self.ih
        self.dict["lx"] = self.lx
        self.dict["ly"] = self.ly
        self.dict["crin"] = self.crin
        self.dict["crout"] = self.crout
        self.dict["CaliAngle"] = self.CaliAngle
        jsonString = js.dumps(self.dict)
        jsonFile = open(rcp + '.json', "w")
        jsonFile.write(jsonString)
        jsonFile.close()
    def saveTemplate(self):
        if self.sourceColor is not None:
            tmpZone = self.sourceColor[int(self.iy + self.cy - self.ch / 2):int(self.iy + self.cy + self.ch / 2), int(self.ix + self.cx - self.cw / 2):int(self.ix + self.cx + self.cw / 2)]
            cv2.imwrite("./Recipe/" + self.rcp + ".jpg" , tmpZone)
            self.tmp = tmpZone.copy()
    def saveLastRecipe(self):
        rcp = "./Recipe/LastRcp/"
        rcp+= self.last_rcp["rcp"]
        jsonString = js.dumps(self.Last_rcp)
        jsonFile = open(rcp + '.json', "w")
        jsonFile.write(jsonString)
        jsonFile.close()
    def cv_imread(self, file_path):
        cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
        return cv_img

    def analysis_img_file(self, file, filetype=0):

        if filetype == 1:
            self.processVedio(file)
        else:
            self.sourceColor = self.cv_imread(file)
            self.proceeImage(self.sourceColor, self.assignedAngle)
            # pygame.init()
            # start = pygame.time.get_ticks()
            # print(str(pygame.time.get_ticks() - start) + "  milliseconds used")
    def proceeImage(self , image , Angle):
        if self.sourceColor is not None:
            self.imageProcee(self.sourceColor, self.assignedAngle)

    def display_video_extract(self, vidcapture, index=0):
        vidcapture.set(cv2.CAP_PROP_POS_FRAMES, index)
        success, self.sourceColor = vidcapture.read()
        self.resourceColor = cv2.resize(self.sourceColor, (1920, 1080), interpolation=cv2.INTER_LINEAR)

    def analysis_vedio(self, sourceColor, index=0):
        z = 0
        self.imageProcee(sourceColor, self.assignedAngle)
        # key = cv2.waitKey()
        # if key == 27:
        #     vedioExtracting = Falsee
        #     cv2.imwrite("p" + str(z) + ".jpg", self.sourceColor)

    resultAngle = 0
    def imageProcee(self, sourceColor = None , assignedAngle = None):
        try:
            pygame.init()
            shared_resource_lock.acquire()
            shared_resource_with_lock = 1
            resourceColor = sourceColor.copy()
            self.resourceColor = sourceColor.copy()
            self.sourceColor = sourceColor.copy()
            shared_resource_lock.release()
            shared_resource_with_lock = 0
            if self.uiloading:
                return
            start = pygame.time.get_ticks()
            maxid = -1
            maxArea = -1
            i = -1
            cnt = 0
            ptx = 0
            pty = 0
            colorChange = False
            preAngle = [0, 0]
            AngleList = []
            allAngleList = []
            ptlist = []
            allList = []
            longList = []
            pxy = []
            # tmp_gray = cv2.cvtColor(self.tmp, cv2.COLOR_BGR2GRAY)

            """comment
            there is formula rotate a dynamic point around a baic point as follows
            To rotate point p1 = (x1, y1) around p (x0, y0) by angle a:
    
            x2 = ((x1 - x0) * cos(a)) - ((y1 - y0) * sin(a)) + x0;
            y2 = ((x1 - x0) * sin(a)) + ((y1 - y0) * cos(a)) + y0;
            where (x2, y2) is the new location of point p1
            (250 , 190 ) w = 140 h = 400
            """

            self.cx = ((self.stdx - (self.stdx)) * math.cos(self.maskAngle * math.pi / 180)) - (
                    (self.stdy - (self.stdy + self.ch / 2)) * math.sin(self.maskAngle * math.pi / 180)) + self.stdx
            self.cy = ((self.stdx - (self.stdx)) * math.sin(self.maskAngle * math.pi / 180)) + (
                    (self.stdy - (self.stdy + self.ch / 2)) * math.cos(self.maskAngle * math.pi / 180)) + (
                              self.stdy + self.ch / 2)
            rotrec = (
                (self.cx + self.ix, self.cy + self.iy),  # center pt
                (self.cw, self.ch),  # W, H
                self.maskAngle  # angle
            )
            realrotrec = (
                (self.cx, self.cy),  # center pt
                (self.cw, self.ch),  # W, H
                self.maskAngle  # angle
            )
            resourceColor = sourceColor.copy()
            source = cv2.cvtColor(sourceColor, cv2.COLOR_BGR2GRAY)
            # (R, source, B) = cv2.split(self.sourceColor)
            resource = cv2.resize(source, (1920, 1080), interpolation=cv2.INTER_LINEAR)
            resourceColor = cv2.circle(resourceColor, (self.lx + self.ix, self.ly + self.iy), self.crin, (255, 0, 0), 2)
            resourceColor = cv2.circle(resourceColor, (self.lx + self.ix, self.ly + self.iy), self.crout, (255, 0, 0),3)
            protractorcenter = [int(self.stdx) + self.ix, int(self.stdy + self.iy + self.ch / 2)]
            for j in range(181):
                valueOffsetY = 400 * math.sin(-1 * j * math.pi / 180)
                valueOffsetX = 400 * math.cos(-1 * j * math.pi / 180)
                if j % 10 == 0:
                    resourceColor = cv2.line(resourceColor, (protractorcenter[0], protractorcenter[1]),
                                                  (protractorcenter[0] + int(valueOffsetX),
                                                   protractorcenter[1] + int(valueOffsetY)), (255, 255, 0), 1)
                    resourceColor = cv2.putText(resourceColor, str(j),
                                                     (protractorcenter[0] + int(valueOffsetX),
                                                      protractorcenter[1] + int(valueOffsetY)),
                                                     cv2.FONT_ITALIC,
                                                     0.5,
                                                     (255, 255, 0), 1, cv2.LINE_AA, False)
            resourceColor = cv2.rectangle(resourceColor, (self.ix, self.iy), (self.ix + self.iw, self.iy + self.ih),
                                          (0, 255, 0), 2)
            shared_resource_lock.acquire()
            shared_resource_with_lock = 1
            self.resourceColor = resourceColor.copy()
            shared_resource_lock.release()
            shared_resource_with_lock = 0
            out = cv2.boxPoints(realrotrec)
            output = cv2.boxPoints(rotrec)
            contour = np.array(output).reshape((-1, 1, 2)).astype(np.int32)
            realcontour = np.array(out).reshape((-1, 1, 2)).astype(np.int32)
            convexHull = cv2.convexHull(contour)
            convexReal = cv2.convexHull(realcontour)
            resourceColor = cv2.drawContours(resourceColor, [convexHull], -1, (0, 0, 255), 1)
            resourceColor = cv2.drawContours(resourceColor, [convexReal], -1, (0, 0, 255), 1)
            # blur = cv2.medianBlur(resource, 7)
            HeadZone = resource[self.iy:(self.iy + self.ih), self.ix:(self.ix + self.iw)]
            blur = cv2.GaussianBlur(HeadZone, (3, 3), 0)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(11, 11))
            erode = cv2.erode(blur, kernel)
            resource = cv2.dilate(erode, kernel)
            edged = cv2.Canny(resource, self.cannyValMin, self.cannyValMax, apertureSize=3, L2gradient=True)
            #cv2.imshow("Canny" , edged)
            h, w = HeadZone.shape
            integerarray = np.array(out.astype(int))
            maskimg = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(maskimg, integerarray, 255)
            result = cv2.bitwise_and(edged, edged, mask=maskimg)
            # =====================================
            if self.tmp is not None:
                hh, ww = self.tmp.shape[0], self.tmp.shape[1]
                HeadZoneMatch = self.sourceColor[self.iy:(self.iy + self.ih), self.ix:(self.ix + self.iw)]
                matched = cv2.matchTemplate(HeadZoneMatch ,self.tmp, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matched)
                cv2.rectangle(resourceColor, (max_loc[0] + self.ix, max_loc[1] + self.iy), (max_loc[0] + ww + self.ix, max_loc[1] + self.iy + hh), (0, 255, 255), 2)
                self.maxPercent = round(max_val * 100.0 , 1)
            # ==============================================================================================
            hh, ww = source.shape
            hh2 = hh // 2
            ww2 = ww // 2

            # define circles
            radius1 = 25
            radius2 = 75
            xc = hh // 2
            yc = ww // 2

            # draw filled circles in white on black background as masks
            # integerarray = np.array(output.astype(int))
            # maskimg = np.zeros((hh, ww), dtype=np.uint8)
            # mask1 = np.zeros_like(img)
            # mask1 = cv2.circle(mask1, (xc, yc), radius1, (255, 255, 255), -1)
            # mask2 = np.zeros_like(img)
            # mask2 = cv2.circle(mask2, (xc, yc), radius2, (255, 255, 255), -1)
            #
            # maskimg = np.zeros((hh, ww), dtype=np.uint8)
            # # subtract masks and make into single channel
            # mask = cv2.subtract(mask2, mask1)
            #
            # # put mask into alpha channel of input
            # result = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            # result[:, :, 3] = mask[:, :, 0]
            # ==============================================================================================

            # kernel = np.array([[0, -1, 0],
            #                    [-1, 5, -1],
            #                    [0, -1, 0]])
            # sharpened = cv2.filter2D(HeadZone, -1, kernel)
            contours, hierarchy = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # contoursRod, hierarchyRod = cv2.findContours(edgedRod, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            # Sort Head
            xx = 0
            yy = 0
            colorIndex = 0


            print("d1")
            length_list = []
            for (i, c) in enumerate(contours):
                colorIndex += 1
                (x, y, w, h) = cv2.boundingRect(c)
                length = math.sqrt(w * w + h * h)
                length_list.append([i, length])
            if len(length_list) < 1:
                shared_resource_lock.acquire()
                self.resourceColor = resourceColor.copy()
                shared_resource_lock.release()
                return
            m = 0
            for contour in contours:
                cnt = 0
                for point in contour:
                    pt = point
                    pt0 = pt[0]
                    if cnt % 10 == 0:
                        ptx = pt0[0]
                        pty = pt0[1]
                    if cnt % 10 == 9:
                        colorChange = not colorChange
                        Vx = pt0[0] - ptx
                        Vy = pt0[1] - pty
                        Vt = math.sqrt(Vx * Vx + Vy * Vy)
                        if Vt == 0:
                            print("e1")
                            continue
                        Angle = math.acos(float(Vx) / float(Vt))
                        Angle = round(Angle * 180 / math.pi, 2)
                        if math.fabs(Angle - preAngle[0]) < 20 and Vt < 14.142135624 and Vy < 0:  # and math.fabs(Angle - preAngle[1]) < 15:#and Vy > 0:
                            #
                            for pidx in range(cnt - 9, cnt):
                                if pidx >= len(contour):
                                    continue
                                pp = contour[pidx]
                                ptlist.append(pp)
                            # if pt0[0] > 0 and pt0[0] < self.iw and pt0[1] > 0 and pt0[1] < self.ih:
                            #   value = HeadZone[pt0[1],pt0[0]]
                            ptlist.append(np.array([[pt0[0], pt0[1]]]))
                            if cnt + 10 >= len(contour):
                                allList.append(ptlist)
                                ptlist = []
                        else:
                            if len(ptlist) > 2:
                                allList.append(ptlist)
                                ptlist = []
                        preAngle[1] = preAngle[0]
                        preAngle[0] = Angle
                    cnt = cnt + 1
            if len(allList) == 0:
                self.state = 3
                shared_resource_lock.acquire()
                shared_resource_with_lock = 1
                self.resourceColor = resourceColor.copy()
                shared_resource_lock.release()
                shared_resource_with_lock = 0
                return
            length_list = []
            topHeight = 9999
            AngleList = []
            print("d2")
            if True:
                for longList in allList:
                    if len(longList) == 0:
                        length_list.append(0.0)
                    if len(longList) > 0:
                        x1 = longList[0][0][0]
                        x2 = longList[len(longList) - 1][0][0]
                        y1 = longList[0][0][1]
                        y2 = longList[len(longList) - 1][0][1]
                        Vx = x2 - x1
                        Vy = y2 - y1
                        Vt = math.sqrt(Vx * Vx + Vy * Vy)
                        if Vt == 0:
                            print("e2")
                            continue
                        Angle = math.acos(Vx / Vt)
                        Angle = round(Angle * 180 / math.pi, 2)
                        length_list.append(Vt)
                        count = 0
                        for xy in longList:
                            count += 1
                            if count == 1:
                                pxy = xy
                                continue
                            Vx = xy[0][0] - pxy[0][0]
                            Vy = xy[0][1] - pxy[0][1]
                            Vt = math.sqrt(Vx * Vx + Vy * Vy)
                            pxy = xy
                    if len(length_list) == 0:
                        continue
                x = np.array(length_list)
                indices = np.argsort(x, axis=0)
                maxid = indices[len(indices) - 1]
                longList = allList[maxid]
                cnt = 0
                print("d3")
                for xy in longList:
                    if cnt == 0:
                        pt = xy
                        ptx = pt[0][0]
                        pty = pt[0][1]
                        cnt += 1
                        pxy = xy
                        continue
                    cnt += 1
                    pt0 = xy[0]
                    Vx = pt0[0] - ptx
                    Vy = pt0[1] - pty
                    Vt = math.sqrt(Vx * Vx + Vy * Vy)
                    if Vt == 0:
                        print("e3")
                        continue
                    Angle = math.acos(float(Vx) / float(Vt))
                    Angle = round(Angle * 180 / math.pi, 2)
                    AngleList.append(Angle)
                    resourceColor = cv2.line(resourceColor,
                                                  (xy[0][0] + self.ix, xy[0][1] + self.iy),
                                                  (pxy[0][0] + self.ix, pxy[0][1] + self.iy),
                                                 (0, 255, 255), 1)
                    pxy = xy
                print("d5")
                if len(longList) > 0:
                    x1 = longList[0][0][0]
                    x2 = longList[len(longList) - 1][0][0]
                    y1 = longList[0][0][1]
                    y2 = longList[len(longList) - 1][0][1]
                    if y2 > y1:
                        Vx = x1 - x2
                    else:
                        Vx = x2 - x1
                    Vy = y2 - y1
                    Vt = math.sqrt(Vx * Vx + Vy * Vy)
                    if Vt == 0:
                        print("e4")
                    Angle = math.acos(float(Vx) / float(Vt))
                    Angle = round(Angle * 180 / math.pi, 2)
                    # if self.isCali:
                    #     self.CaliAngle = -Angle
                    # Angle = Angle - CaliAngle
                    # Angle = round(Angle, 2) + self.CaliAngle
                    m = 0
                    for pt in longList:
                        longList[m][0][0] += self.ix
                        longList[m][0][1] += self.iy
                        m = m + 1
                    start = pygame.time.get_ticks() - start
                    print("access time = " + str(start))
                    ctr = np.array(longList).reshape((-1, 1, 2)).astype(np.int32)
                    if len(ctr) > 5:
                        rect = cv2.minAreaRect(ctr)
                        fltAngle = round(rect[2], 2)
                        if fltAngle > 45:
                            fltAngle -= 90
                        else:
                            pass
                        box = cv2.boxPoints(rect)
                        resourceColor = cv2.line(resourceColor, (int(box[0][0]), int(box[0][1])),
                                                                                            (int(box[1][0]), int(box[1][1]))
                                                                                             , (255, 255, 0), 1)
                        resourceColor = cv2.line(resourceColor, (int(box[1][0]), int(box[1][1])),
                                                      (int(box[2][0]), int(box[2][1]))
                                                      , (255, 255, 0), 1)
                        resourceColor = cv2.line(resourceColor, (int(box[2][0]), int(box[2][1])),
                                                      (int(box[3][0]), int(box[3][1]))
                                                      , (255, 255, 0), 1)
                        resourceColor = cv2.line(resourceColor, (int(box[3][0]), int(box[3][1])),
                                                      (int(box[0][0]), int(box[0][1]))
                                                      , (255, 255, 0), 1)

                        xy1 = [[0,0]]
                        xy2 = [[0,0]]
                        idxc = 0
                        idyc = 0
                        idx = 0
                        idy = 0
                        cnt = 0
                for xy in longList:
                    pt0 = xy[0]
                    Vx = pt0[0] - (self.lx + self.ix)
                    Vy = pt0[1] - (self.ly + self.iy)
                    Vt = math.sqrt(Vx * Vx + Vy * Vy)
                    if Vt > self.crin - 2 and Vt < self.crin + 2:
                        xy1 += xy
                        idxc += 1
                        idx = cnt
                    if Vt > self.crout - 2 and Vt < self.crout + 2:
                        xy2 += xy
                        idyc += 1
                        idy = cnt
                    cnt += 1

                if idxc == 0 or idyc == 0:
                    self.state = 4
                    print("e6")
                    shared_resource_lock.acquire()
                    shared_resource_with_lock = 1
                    self.resourceColor = resourceColor.copy()
                    shared_resource_lock.release()
                    shared_resource_with_lock = 0
                    return
                dx = 10 - idxc
                dy = 10 - idyc
                print("d6")
                if idx + dx < len(longList) - 1 and idy + dy < len(longList) - 1:
                    for i in range(idx, idx + dx):
                        xy1 += longList[i]
                        idxc += 1
                    for i in range(idy, idy + dy):
                        xy2 += longList[i]
                        idyc += 1
                print("d6.6")
                if idxc == 0 or idyc == 0:
                    print("e5")
                    shared_resource_lock.acquire()
                    shared_resource_with_lock = 1
                    self.resourceColor = resourceColor.copy()
                    shared_resource_lock.release()
                    shared_resource_with_lock = 0
                    return
                print("d7")
                xy1 = xy1 / idxc
                xy2 = xy2 / idyc
                resourceColor = cv2.line(resourceColor, (int(xy1[0][0]) , int(xy1[0][1])),
                                              (int(xy2[0][0]), int(xy2[0][1]))
                                              , (0, 255, 0), 1)
                Vx = xy2[0][0] - xy1[0][0]
                Vy = xy2[0][1] - xy1[0][1]
                Vt = math.sqrt(Vx * Vx + Vy * Vy)
                if Vt == 0:
                    print("e6")
                    shared_resource_lock.acquire()
                    self.resourceColor = resourceColor.copy()
                    shared_resource_lock.release()
                    return
                Angle = math.acos(float(Vx) / float(Vt))
                Angle = Angle * 180 / math.pi
                Angle = round(90.0 - Angle, 1)
                strAngle = str(Angle)
                phi = (90.0 - Angle) * math.pi / 180
                self.resultAngle = Angle
                if self.assignedAngle - 0.5 < Angle < self.assignedAngle + 0.5:
                    self.state = 0
                    resourceColor = cv2.putText(resourceColor, strAngle + " degree" + " OK",
                                                     (self.ix, self.iy),
                                                     cv2.FONT_ITALIC,
                                                     1,
                                                     (0, 255, 0), 2, cv2.LINE_AA, False)
                else:
                    self.state = 1
                    if Angle < -3 or Angle > 3:
                        self.state = 2
                    resourceColor = cv2.putText(resourceColor, strAngle + " degree" + " NG",
                                                     (self.ix, self.iy),
                                                     cv2.FONT_ITALIC,
                                                     1,
                                                     (0, 0, 255), 2, cv2.LINE_AA, False)
                    if Angle > self.assignedAngle :
                        strArrow = str(math.fabs(round(Angle - self.assignedAngle , 1)))

                        resourceColor = cv2.putText(resourceColor, "<<<<=" + strArrow,
                                                         (self.ix, self.iy - 50),
                                                         cv2.FONT_ITALIC,
                                                         3,
                                                     (0, 0, 255), 2, cv2.LINE_AA, False)
                    else:
                        strArrow = str(math.fabs(round(Angle - self.assignedAngle, 1)))

                        resourceColor = cv2.putText(resourceColor, "=>>>>" + strArrow,
                                                         (self.ix, self.iy - 50),
                                                         cv2.FONT_ITALIC,
                                                         3,
                                                         (0, 0, 255), 2, cv2.LINE_AA, False)
                resourceColor = cv2.arrowedLine(resourceColor,
                                                     (int(self.stdx + self.ix), int(self.stdy + self.iy + self.ch / 2)),
                                                     (int(350 * math.cos(phi) + self.stdx + self.ix),
                                                      int(-350 * math.sin(
                                                          math.fabs(phi)) + self.stdy + self.iy + self.ch / 2)),
                                                     (255, 0, 255), 1)
                print("d8")
                shared_resource_lock.acquire()
                shared_resource_with_lock = 1
                self.resourceColor = resourceColor.copy()
                shared_resource_lock.release()
                shared_resource_with_lock = 0
                print("d9")
        except:
            print("shared_resource_with_lock = " + str(shared_resource_with_lock))
            if shared_resource_with_lock == 1:
                shared_resource_lock.release()
            print("al error")
