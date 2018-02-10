from __future__ import division
import os
os.environ['THEANO_FLAGS'] = "device=cpu"

import cv2
import numpy as np
from keras.models import load_model
import math
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

global konacni_rezultat
epsilon = 1e-12

global k1,l1,k2,l2
k1,l1,k2,l2=None,None,None,None
global n1,m1,n2,m2
n1,m1,n2,m2=None,None,None,None

global p1,p2,p3,p4
p1,p2,p3,p4 = None,None,None,None

global width, height

diff = 15
font = cv2.FONT_HERSHEY_SIMPLEX

global global_roi_list

class Point:
	def __init__(self,x,y):
		self.x = x
		self.y = y

class Roi(object):
    def __init__(self, params):
        self.x = params.get('x')
        self.y = params.get('y')
        self.center = params.get('center')
        self.br_frejma = params.get('br_frejma')
        self.id = params.get('id')
        self.region = params.get('region')
        self.predicted = params.get('predicted')
        self.probability = params.get('probability')
        self.passed_first = params.get('passed_first')
        self.passed_second = params.get('passed_second')
        self.path = params.get('path')
        self.path_vector = params.get('path_vector')
        self.width = params.get('width')
        self.height = params.get('height')

def find_rois(image_orig, image_bin, model, br_frejma):
    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rois_in_current_frame = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)  # koordinate i velicina granicnog pravougaonika
        area = cv2.contourArea(contour)
        #if area > 20 and h < 30 and h > 9 and w > 0: #ovo radi proverno
        if x-4>=0 and y - 4>=0:
            if h < 30 and h > 7 and w > 0:
                if checkIfLoadedFully(x,y):
                    region = image_bin[y - 4:y + h + 4, x - 4:x + w + 4]

                    #broj_belih = cv2.countNonZero(region)
                    # if broj_belih > 165:
                    #      region = erode(region)
                    # elif broj_belih < 75:
                    #      region = dilate(region)

                    region = resize_region(region)  #skaliranje na 28x28=784

                    #####################CNN######################
                    region = region.reshape(1, 28, 28, 1).astype('float32') #kompatibilan oblik za mrezu
                    region = region / 255 #normalizuj
                    rez = model.predict(region, verbose=0)
                    cv2.putText(image_orig, str(np.argmax(rez)), (x + 2, y + 2), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.putText(image_orig, str(np.max(rez)), (x + 20, y + 1), font, 0.4, (0, 255, 255), 1, cv2.LINE_AA) #%
                    ###############################################

                    c1 = (x + x + w) / 2
                    c2 = (y + y + h) / 2
                    center = Point(c1, c2)

                    #cv2.putText(image_orig, str(center.y), (x + 2, y + 2), font, 1, (0, 0, 255), 1, cv2.LINE_AA)

                    values = {"x": x,
                              "y": y,
                              "center": center,
                              "br_frejma": br_frejma,
                              "id": -1,
                              "region": region,
                              "predicted": np.argmax(rez),
                              "probability": np.max(rez),
                              "passed_first": False,
                              "passed_second": False,
                              "height": h,
                              "width": w,
                              "path": [],
                              "path_vector": None}

                    novi_obj = Roi(values)

                    rois_in_current_frame.append(novi_obj)

                    #draw

                    #okvir
                    cv2.rectangle(image_orig, (x-4, y-4), (x + w + 4, y + h + 4), (0, 255, 0), 1)
                    #centar roi
                    #cv2.rectangle(image_orig, (int(novi_obj.center.x), int(novi_obj.center.y)),(int(novi_obj.center.x) + 1, int(novi_obj.center.y) + 1), (255, 0, 255), 2)

                    #dijagonala
                    #cv2.line(image_orig, (x,y),(x + 20,y + 20), (255, 0, 255), 1)

    #dodaje novi ili samo update-je vec postojece
    updateCenterOrAddNew(rois_in_current_frame)

    #ako neki nestane
    lastCheck(rois_in_current_frame, image_orig)

    return image_orig, rois_in_current_frame

def ccw(A,B,C):
    return (C.y-A.y) * (B.x-A.x) > (B.y-A.y) * (C.x-A.x)

def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

#ovako se globalna lista krece
def updateCenterOrAddNew(elements):
    for el in elements:
        exists = False
        global global_roi_list
        for gl_el in global_roi_list:
            if math.hypot(gl_el.center.x - el.center.x, gl_el.center.y - el.center.y) < diff:
                #gl_el = el
                #samo update
                gl_el.center.x = el.center.x
                gl_el.center.y = el.center.y
                gl_el.x = el.x
                gl_el.y = el.y
                gl_el.path.append((el.center.x, el.center.y))

                # #ako se vrednosti ne poklapaju
                if gl_el.predicted != el.predicted:
                    #i ako je verovatnoca veca
                    if gl_el.probability < el.probability:
                        if gl_el.passed_first is False and gl_el.passed_second is False:
                            #samo ako jos nije blizu linije
                            if gl_el.center.x < p1.x - 100 and gl_el.center.y < p1.y - 100 and gl_el.center.x < p2.x - 100 and gl_el.center.y < p2.y - 100:
                                print 'izmena'
                                gl_el.predicted = el.predicted


                exists = True
                break
        if not exists:
            if len(global_roi_list) == 0:
                el.id = 0
            else:
                el.id = len(global_roi_list)
            global_roi_list.append(el)

def aboveOrBelow(point,start,end):
    v1 = (end.x - start.x, end.y - start.y)  # Vektor 1
    v2 = (end.x - point.x, end.y - point.y)  # Vektor 2
    xp = v1[0] * v2[1] - v1[1] * v2[0] # Cross product

    if xp > 0:
        return 'iznad'
    elif xp < 0:
        return 'ispod'
    else:
        return 'same'

def checkIfLoadedFully(x,y):
    #return x - 30 > 0 and x < width + 30 and y - 30 > 0 and y < height + 30
    return True

def isInScene(el):
    return el.center.x > 0 and el.center.x < width and el.center.y > 0 and el.center.y < height

def intersection(img):
    for gl_el in global_roi_list:

        presek_centra_i_prve_prave = None
        presek_centra_i_druge_prave = None

        if p1 is not None and p2 is not None:
            presek_centra_i_prve_prave = intersect(p1,p2,(Point(gl_el.x,gl_el.y)),Point(gl_el.x-25,gl_el.y-25))
        if p3 is not None and p4 is not None:
            presek_centra_i_druge_prave = intersect(p3, p4, (Point(gl_el.x, gl_el.y)), Point(gl_el.x - 25, gl_el.y -25))

        if presek_centra_i_prve_prave is not None:
            if (presek_centra_i_prve_prave) and not gl_el.passed_first:
                gl_el.passed_first = True
                print("+++Presek prve prave sa brojem:+++ {}".format(gl_el.predicted))
                cv2.circle(img, (int(gl_el.center.x),int(gl_el.center.y)), 16, (255,0,255), 2)
                global konacni_rezultat
                konacni_rezultat += gl_el.predicted
                print("[KONACNI REZULTAT: {}]".format(konacni_rezultat))
        if presek_centra_i_druge_prave is not None:
            if (presek_centra_i_druge_prave) and not gl_el.passed_second:
                gl_el.passed_second = True
                print("---Presek druge prave sa brojem:--- {}".format(gl_el.predicted))
                cv2.circle(img, (int(gl_el.center.x),int(gl_el.center.y)), 16, (25,0,255), 2)
                global konacni_rezultat
                konacni_rezultat -= gl_el.predicted
                print("[KONACNI REZULTAT: {}]".format(konacni_rezultat))

#poslednja provera, ako su neke cifre vidjene u celini samo jedanput
#proverava kada je poslednji put vidjen neki region
#ako mu je zadnja zabelezena pozicija bila unutar kadra,
#a sada ga nema, znaci da je preklopljen
#nalazi njegovu putanju i azurira rezultat ako je potrebno
def lastCheck(curr_elements, img):
    alive = False
    for gl_el in global_roi_list:
        for el in curr_elements:
            if gl_el.id == el.id:
                alive = True
                break
        if len(gl_el.path)>0:
            if not alive and gl_el.path[-1][0] + 15< width and gl_el.path[-1][1] +15 < height:
                checkOnceMore(gl_el, img)

def checkOnceMore(gl_el,img):
    x = []
    y = []

    for point in gl_el.path:
        x.append(point[0])
        y.append(point[1])

    # ako je presao bar 50 tacaka
    if len(x) > 0:
        poc_poz = aboveOrBelow(Point(gl_el.path[0][0], gl_el.path[0][1]), p1, p2)
        tre_poz = aboveOrBelow(gl_el.center, p1, p2)

        if poc_poz == 'iznad':
            if tre_poz == 'ispod':
                if not gl_el.passed_first:
                    fit = np.polyfit(x, y, 1)
                    f = np.poly1d(fit)

                    result = findIntersection(f, fja, 0.0)

                    # ako je po putanji presekao
                    if len(result) > 0:
                        # i ako je presek u okviru X vrednosti prve prave
                        if result[0] > p1.x and result[0] < p2.x:
                            #######plot######################
                            # xs1 = np.linspace(p1.x, p2.x, 100)
                            # ys1 = np.linspace(p1.y, p2.y, 100)
                            # xs2 = np.linspace(p3.x, p4.x, 100)
                            # ys2 = np.linspace(p3.y, p4.y, 100)
                            # axes = plt.gca()
                            # axes.set_xlim([0, width])
                            # axes.set_ylim([0, height])
                            # plt.gca().invert_yaxis()
                            # plt.plot(xs1, fja(xs1), xs1, f(xs1), result, f(result), 'ro')
                            #
                            # plt.plot(x, y, xs1, ys1, xs2, ys2)
                            # plt.xlabel('Presek prve prave sa putanjom cifre')
                            # plt.show()
                            #################################

                            gl_el.passed_first = True
                            print("LAST KOR: PLUS KOREKCIJA:+++ {}".format(gl_el.predicted))
                            cv2.circle(img, (int(gl_el.path[0][0]), int(f(gl_el.path[0][0]))), 16, (0, 0, 255), 2)
                            cv2.line(img, (int(gl_el.path[0][0]), int(gl_el.path[0][1])),
                                     (int(gl_el.center.x), int(gl_el.center.y)), (0, 0, 255), 1)
                            cv2.circle(img, (int(gl_el.center.x), int(gl_el.center.y)), 16, (0, 0, 255), 2)
                            global konacni_rezultat
                            konacni_rezultat += gl_el.predicted
                            print("[KONACNI REZULTAT: {}]".format(konacni_rezultat))



        poc_poz = aboveOrBelow(Point(gl_el.path[0][0], gl_el.path[0][1]), p3, p4)
        tre_poz = aboveOrBelow(Point(gl_el.path[-1][0], gl_el.path[-1][1]), p3, p4)

        if poc_poz == 'iznad':
            if tre_poz == 'ispod':

                if not gl_el.passed_second:
                    fit = np.polyfit(x, y, 1)
                    f = np.poly1d(fit)

                    result = findIntersection(f, fja2, 0.0)
                    result_drugi = intersect(p3, p4, (Point(gl_el.path[0][0], f(gl_el.path[0][0]))), Point(gl_el.path[-1][0], f(gl_el.path[-1][1])))
                    cv2.line(img, (int(gl_el.path[0][0]), int(gl_el.path[0][1])),
                             (int(gl_el.path[-1][0]), int(gl_el.path[-1][1])), (0, 0, 255), 1)

                    # ako je po putanji presekao
                    if len(result) > 0:
                        # i ako je presek u okviru X vrednosti prave
                        if (result[0] > p3.x and result[0] < p4.x):
                            gl_el.passed_second = True
                            print("LAST COR: MINUS KOREKCIJA:--- {}".format(gl_el.predicted))
                            cv2.circle(img, (int(gl_el.path[0][0]), int(f(gl_el.path[0][0]))), 16, (0, 0, 255), 2)
                            cv2.line(img, (int(gl_el.path[0][0]), int(gl_el.path[0][1])),
                                     (int(gl_el.center.x), int(gl_el.center.y)), (0, 0, 255), 1)
                            cv2.circle(img, (int(gl_el.center.x), int(gl_el.center.y)), 16, (0, 0, 255), 2)
                            global konacni_rezultat
                            konacni_rezultat -= gl_el.predicted
                            print("[KONACNI REZULTAT: {}]".format(konacni_rezultat))

                            #######plot########################
                            # xs1 = np.linspace(p1.x, p2.x, 100)
                            # ys1 = np.linspace(p1.y, p2.y, 100)
                            # xs2 = np.linspace(p3.x, p4.x, 100)
                            # ys2 = np.linspace(p3.y, p4.y, 100)
                            #
                            # axes = plt.gca()
                            # axes.set_xlim([0, width])
                            # axes.set_ylim([0, height])
                            # plt.gca().invert_yaxis()
                            #
                            # plt.plot(xs2, fja2(xs2), xs2, f(xs2), result, f(result), 'ro')
                            # plt.plot(x, y, xs1, ys1, xs2, ys2)
                            # plt.xlabel('Presek druge prave sa putanjom cifre')
                            # plt.show()
                            ###################################

#proverava preklapanje ako se cifra pojavi ispod linije
#ne proverava ako je cifra preklopljena sve do izlaska iz kadra
def checkOverlappingCurveFitting(img):
    for gl_el in global_roi_list:
        x = []
        y = []

        for point in gl_el.path:
            x.append(point[0])
            y.append(point[1])

        #ako je presao bar 50 tacaka
        if len(x) > 0:
            poc_poz = aboveOrBelow(Point(gl_el.path[0][0],gl_el.path[0][1]),p1,p2)
            tre_poz = aboveOrBelow(Point(gl_el.path[-1][0], gl_el.path[-1][1]),p1,p2)

            if poc_poz == 'iznad':
                if tre_poz == 'ispod':
                    if not gl_el.passed_first:
                        fit = np.polyfit(x, y, 1)
                        f = np.poly1d(fit)

                        result = findIntersection(f, fja, 0.0)

                        #ako je po putanji presekao
                        if len(result) > 0:
                            #i ako je presek u okviru X vrednosti prve prave
                            if result[0] > p1.x and result[0] < p2.x:

                                gl_el.passed_first = True
                                print("PLUS KOREKCIJA:+++ {}".format(gl_el.predicted))
                                cv2.circle(img, (int(gl_el.path[0][0]), int(f(gl_el.path[0][0]))), 16, (0, 0, 255), 2)
                                cv2.line(img, (int(gl_el.path[0][0]), int(gl_el.path[0][1])), (int(gl_el.center.x), int(gl_el.center.y)), (0, 0, 255), 1)
                                cv2.circle(img, (int(gl_el.center.x), int(gl_el.center.y)), 16, (0, 0, 255), 2)

                                global konacni_rezultat
                                konacni_rezultat += gl_el.predicted
                                print("[KONACNI REZULTAT: {}]".format(konacni_rezultat))


            poc_poz = aboveOrBelow(Point(gl_el.path[0][0], gl_el.path[0][1]), p3, p4)
            tre_poz = aboveOrBelow(Point(gl_el.path[-1][0], gl_el.path[-1][1]), p3, p4)

            if poc_poz == 'iznad':
                if tre_poz == 'ispod':
                    #a nije je presao
                    if not gl_el.passed_second:

                        fit = np.polyfit(x, y, 1)
                        f = np.poly1d(fit)

                        result = findIntersection(f,fja2,0.0)
                        result_drugi = intersect(p3, p4, (Point(gl_el.path[0][0], f(gl_el.path[0][0]))),
                                                 Point(gl_el.path[-1][0], f(gl_el.path[-1][1])))
                        cv2.line(img, (int(gl_el.path[0][0]), int(gl_el.path[0][1])),
                                 (int(gl_el.path[-1][0]), int(gl_el.path[-1][1])), (0, 0, 255), 1)
                        #ako je po putanji presekao
                        if len(result) > 0:# or result_drugi:
                            #i ako je presek u okviru X vrednosti prave
                            if result[0] > p3.x and result[0] < p4.x:

                                gl_el.passed_second = True
                                print("MINUS KOREKCIJA:--- {}".format(gl_el.predicted))
                                cv2.circle(img, (int(gl_el.path[0][0]), int(f(gl_el.path[0][0]))), 16, (0, 0, 255), 2)
                                cv2.line(img, (int(gl_el.path[0][0]), int(gl_el.path[0][1])), (int(gl_el.center.x), int(gl_el.center.y)), (0, 0, 255), 1)
                                cv2.circle(img, (int(gl_el.center.x), int(gl_el.center.y)), 16, (0, 0, 255), 2)

                                global konacni_rezultat
                                konacni_rezultat -= gl_el.predicted
                                print("[KONACNI REZULTAT: {}]".format(konacni_rezultat))

#funkcija prve prave
global m, c
def fja(x):
    return m*x+c

#funkcija druge prave
global mm, cc
def fja2(x):
    return mm*x+cc

#presek funkcija
def findIntersection(fun1,fun2,x0):
 return fsolve(lambda x : fun1(x) - fun2(x),x0)

#izracunaj kooficijente pravih
def makeFunction(x,y,w,z):
    from numpy import ones, vstack
    from numpy.linalg import lstsq
    points = [(x, y), (w, z)]
    x_coords, y_coords = zip(*points)
    A = vstack([x_coords, ones(len(x_coords))]).T
    mf, cf = lstsq(A, y_coords)[0]
    #print("Line Solution is y = {m}x + {c}".format(m=m, c=c))
    return mf, cf

def resize_region(region):
    if not region is None:
        return cv2.resize(region,(28,28), interpolation = cv2.INTER_NEAREST)

def dilate(image):
    kernel = np.ones((2,2))
    return cv2.dilate(image, kernel, iterations=1)

def erode(image):
    kernel = np.ones((2,2))
    return cv2.erode(image, kernel, iterations=1)

def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def image_bin(image_gs):
    ret,image_bin = cv2.threshold(image_gs, 127, 255, cv2.THRESH_BINARY)
    return image_bin

def invert(image):
    return 255-image

def obradi_video(video,f):
    global konacni_rezultat
    konacni_rezultat = 0

    global global_roi_list
    global_roi_list = []

    #############CNN Model###############
    # Ako model ne postoji, treniraj
    # model = kerasCNN.train()
    # model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
    # del model
    ######################################
    model = load_model('cnn.h5')
    ######################################

    cap = cv2.VideoCapture('resources/%s' % (video))

    global width
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  #float
    global height
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  #float

    curr_frame_num = 0
    while(cap.isOpened()):
        ret, frame = cap.read()

        if frame is None:
            print 'Greska prilikom ucitavanja frejma ili kraj snimka'
            break

        if curr_frame_num == 0:
            #maska za prvu liniju
            lowerl1 = np.array([230, 0, 0])
            upperl1 = np.array([255, 155, 155])
            maskl1 = cv2.inRange(frame, lowerl1, upperl1)

            frame_for_first_line = frame.copy()
            frame_for_first_line = cv2.bitwise_and(frame_for_first_line, frame_for_first_line, mask=maskl1)

            #maska za drugu liniju
            lowerl2 = np.array([0, 230, 0])
            upperl2 = np.array([155, 255, 155])
            maskl2 = cv2.inRange(frame, lowerl2, upperl2)

            frame_for_second_line = frame.copy()
            frame_for_second_line = cv2.bitwise_and(frame_for_second_line, frame_for_second_line, mask=maskl2)

            grayScale = cv2.cvtColor(frame_for_first_line, cv2.COLOR_BGR2GRAY)
            ret, thresh_img = cv2.threshold(grayScale,0, 255, cv2.THRESH_BINARY)
            kernel = np.ones((3, 3), np.uint8)
            erosion = cv2.erode(thresh_img, kernel, iterations=1)
            line1 = cv2.HoughLinesP(erosion, 1, np.pi / 180, 100, minLineLength=15, maxLineGap=40)

            grayScale = cv2.cvtColor(frame_for_second_line, cv2.COLOR_BGR2GRAY)
            ret, thresh_img = cv2.threshold(grayScale, 4, 58, cv2.THRESH_BINARY)
            kernel = np.ones((3, 3), np.uint8)
            erosion = cv2.erode(thresh_img, kernel, iterations=1)
            line2 = cv2.HoughLinesP(erosion, 1, np.pi / 180, 100, minLineLength=15, maxLineGap=40)

            #sacuvaj pronadjene linije
            if line1 is not None:
                for x1, y1, x2, y2 in line1[0]:
                    global k1, l1, k2, l2
                    k1, l1, k2, l2 = x1, y1, x2, y2
                    global p1, p2
                    p1 = Point(k1, l1)
                    p2 = Point(k2, l2)
                    print (k1, l1), (k2, l2)
            if line2 is not None:
                for x1, y1, x2, y2 in line2[0]:
                    global n1, m1, n2, m2
                    n1, m1, n2, m2 = x1, y1, x2, y2
                    global p3, p4
                    p3 = Point(n1, m1)
                    p4 = Point(n2, m2)
                    print (n1, m1), (n2, m2)

        #nakon sto pronadje linije, erozija, dilatacija
        img = image_bin(image_gray(frame))
        img_bin = erode(dilate(img))

        #pokupi regione
        selected_regions, rois = find_rois(frame.copy(), img_bin, model, curr_frame_num)

        #proveri preseke
        intersection(selected_regions)

        #proveri da li je neki propusten
        checkOverlappingCurveFitting(selected_regions)

        #izracunaj funkcije pronadjenih prava
        if k1 is not None and l1 is not None and k2 is not None and l2 is not None:
            cv2.line(selected_regions, (k1, l1), (k2, l2), (255, 0, 255), 1)
            global m, c
            m, c = makeFunction(p1.x, p1.y, p2.x, p2.y)

        if n1 is not None and m1 is not None and n2 is not None and m2 is not None:
            cv2.line(selected_regions, (n1, m1), (n2, m2), (255, 0, 255), 1)
            global mm, cc
            mm, cc = makeFunction(p3.x, p3.y, p4.x, p4.y)

        #nacrtaj putanju
        for gl_el in global_roi_list:
            br = 0
            if isInScene(gl_el):
                if len(gl_el.path)>12:
                    for point_in_past in gl_el.path:
                        br+=1
                        if br > 200:
                            break
                        cv2.circle(selected_regions, (int(point_in_past[0]), int(point_in_past[1])), 1, (0,255,47), 1)


        #draw end points
        #cv2.rectangle(frame, (p3.x, p3.y), (p3.x+2, p3.y+2), (255, 0, 255), 2)
        #cv2.rectangle(frame, (p4.x, p4.y), (p4.x+2, p4.y+2), (255, 0, 255), 2)

        cv2.rectangle(selected_regions, (15, height - 55), (87, height-15), (255,255,255), -1)
        cv2.putText(selected_regions, str(konacni_rezultat), (18, height - 25), font, 1, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow('%s' % video, selected_regions)

        curr_frame_num += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("{}".format(video))

    print '**********************************'

    f.write('%s\t%d\n' % (video, konacni_rezultat))

    cap.release()
    cv2.destroyAllWindows()

def main():
    f = open('out_new.txt', 'w')
    f.write('RA 111/2014 Danilo Bujisa\n')
    f.write('file	sum\n')
    videos = ['video-0.avi', 'video-1.avi', 'video-2.avi', 'video-3.avi', 'video-4.avi', 'video-5.avi', 'video-6.avi',
         'video-7.avi', 'video-8.avi', 'video-9.avi']
    # videos = ['video-0.avi']
    for video in videos:
        obradi_video(video,f)

    f.close()

    print 'Done'

if __name__ == '__main__':
    main()

#program.py
#Procenat tacnosti: 45.7831325301
#za dist = 15
#fajl out.txt



