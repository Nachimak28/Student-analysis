from PyQt4 import QtCore, QtGui
from PyQt4.QtGui import QMessageBox
import sys
import csv
import numpy as np
import statistics as st
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn.cluster import KMeans


try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)



class MainWindow(QtGui.QMainWindow):
    def __init__(self, parent = None):
        super(MainWindow, self).__init__(parent)
        self.central_widget = QtGui.QStackedWidget()
        self.setCentralWidget(self.central_widget)
        login_widget = LoginWidget(self)
        #e = str(login_widget.emailIDLineEdit.text())
        #p = str(login_widget.passwordLineEdit.text())
        #if e == "nachi" and p == "123":
        #    print("Login successful")
        #    login_widget.pushButton.clicked.connect(self.postLogin)
        #else:
        #    msg = QMessageBox()
        #    msg.setIcon(QMessageBox.Critical)

        #    msg.setText("Incorrect Credentials")
        #    msg.setInformativeText("Please enter correct credentials.")
        #    msg.setWindowTitle("Incorrect Crendentials")
            #msg.setDetailedText("The details are as follows:")
        #    msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        #    retval = msg.exec_()
        login_widget.pushButton.clicked.connect(self.postLogin)
        self.central_widget.addWidget(login_widget)
#    def postLogin(self):
#        print("Next screen")

#    def postLogin(self):
#        e = str(login_widget.emailIDLineEdit.text())
#        p = str(login_widget.passwordLineEdit.text())
#        if e == "nachi" and p == "123":
#            print("Login successful")
#            second_screen = SecondScreen(self)
#            self.central_widget.addWidget(second_screen)
#            self.central_widget.setCurrentWidget(second_screen)
#            second_screen.pushButton.clicked.connect(self.analysis)
#        else:
#            msg = QMessageBox()
#            msg.setIcon(QMessageBox.Critical)

#            msg.setText("Incorrect Credentials")
#            msg.setInformativeText("Please enter correct credentials.")
#            msg.setWindowTitle("Incorrect Crendentials")
            #msg.setDetailedText("The details are as follows:")
#            msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
#            retval = msg.exec_()
            

    def postLogin(self):
        second_screen = SecondScreen(self)
        self.central_widget.addWidget(second_screen)
        self.central_widget.setCurrentWidget(second_screen)
        second_screen.pushButton.clicked.connect(self.analysis)
        

    def analysis(self):
        analyse_file = AnalyseFile(self)
        self.central_widget.addWidget(analyse_file)
        self.central_widget.setCurrentWidget(analyse_file)
        analyse_file.pushButton.clicked.connect(self.analyse1)
        analyse_file.pushButton_2.clicked.connect(self.report1)
        analyse_file.pushButton_3.clicked.connect(self.analyse2)
        analyse_file.pushButton_4.clicked.connect(self.report1)
        analyse_file.pushButton_5.clicked.connect(self.Exit)
        analyse_file.pushButton_6.clicked.connect(self.postLogin)

    def Exit(self):
        sys.exit(0)

    def analyse1(self):
        
        fname = QtGui.QFileDialog.getOpenFileName(self, 'Open file', 'C:/Users/nachiket/Desktop/attendance research correlation',"CSV files (*.csv)")
        print(fname)
        data = []
        a,b,c,d,e,f = fname.split('/')
        sub,ext = f.split('.')
        print(sub)
        #opening and appending data to list
        with open(fname,'r') as csvfile:
            readCSV = csv.reader(csvfile,delimiter=',')
            for row in readCSV:
                data.append(row)
        x = []
        y = []
        #separating lists
        for row in range(len(data)):
            x.append(float(data[row][0]))
            y.append(float(data[row][1]))
        #normalization
        xbar = st.mean(x)
        sx = max(x)-min(x)
        ybar = st.mean(y)
        sy = max(y)-min(y)
        for i in range(len(x)):
            data[i][0]=(x[i]-xbar)/sx
            data[i][1]=(y[i]-ybar)/sy
        #clustering
        X = np.array(data)
        km = KMeans(n_clusters=4,random_state=42)
        km.fit(X)
        centroids = km.cluster_centers_
        labels = km.labels_
        colors = 10*['r.','g.','b.','c.','k.','y.','m.']

        for i in range(len(X)):
            print("coordinate:",X[i],"label:",labels[i])
            plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize = 10)
    
        plt.scatter(centroids[:, 0], centroids[:, 1], marker = 'x', s = 150, linewidths = 5, zorder = 10)
        plt.xlabel("Attendance")
        plt.ylabel("Marks")
        plt.title(sub.upper())
        plt.show()


        #correlation index
        n = len(x)
        xy = []
        for i in range(len(x)):
            xy.append(float(x[i]*y[i]))
        r = (sum(xy) - n*st.mean(x)*st.mean(y))/(n*st.stdev(x)*st.stdev(y))
        print("Correlation index: ",r)
        text = "Correlation index for subject "+sub.upper()+" = "+str(r)
        txtfile = open('Example1.txt','w')
        txtfile.write(text)
        txtfile.close()
        co = [r,0,0,0]

        N = 4
        width = 0.35
        ind = np.arange(N)
        plt.bar(ind,co,width,color='r',bottom=0)
        plt.ylim(0,1)
        plt.title("Correlation Graph for single subjects") 
        plt.show()
        








        
    def analyse2(self):
        fname = QtGui.QFileDialog.getOpenFileName(self, 'Open file', 'C:/Users/nachiket/Desktop/attendance research correlation',"CSV files (*.csv)")
        #self.le.setPixmap(QPixmap(fname))
        print(fname)
        with open(fname,'r') as csvfile:
            readCSV = csv.reader(csvfile,delimiter=',')
            for row in readCSV:
                l = len(row)

        if l ==8:
            with open(fname,'r') as csvfile:
                readCSV = csv.reader(csvfile,delimiter=',')
            
                a1 = []
                m1 = []
                a2 = []
                m2 = []
                a3 = []
                m3 = []
                a4 = []
                m4 = []
                s1 = []
                s2 = []
                s3 = []
                s4 = []
                for row in readCSV:
                    a = row[0]
                    b = row[1]
                    c = row[2]
                    d = row[3]
                    e = row[4]
                    f = row[5]
                    g = row[6]
                    h = row[7]
                    a1.append(float(a))
                    m1.append(float(b))
                    a2.append(float(c))
                    m2.append(float(d))
                    a3.append(float(e))
                    m3.append(float(f))
                    a4.append(float(g))
                    m4.append(float(h))
                for row in range(len(a1)):
                    p = [a1[row],m1[row]]
                    q = [a2[row],m2[row]]
                    r = [a3[row],m3[row]]
                    s = [a4[row],m4[row]]
                    s1.append(p)
                    s2.append(q)
                    s3.append(r)
                    s4.append(s)
            mean1 = st.mean(a1)
            mean2 = st.mean(m1)
            mean3 = st.mean(a2)
            mean4 = st.mean(m2)
            mean5 = st.mean(a3)
            mean6 = st.mean(m3)
            mean7 = st.mean(a4)
            mean8 = st.mean(m4)
            range1 = max(a1)-min(a1)
            range2 = max(m1)-min(m1)
            range3 = max(a2)-min(a2)
            range4 = max(m2)-min(m2)
            range5 = max(a3)-min(a3)
            range6 = max(m3)-min(m3)
            range7 = max(a4)-min(a4)
            range8 = max(m4)-min(m4)
        #normalizing
            for row in range(len(a1)):
                s1[row][0]=(a1[row]-mean1)/range1
                s1[row][1]=(m1[row]-mean2)/range2
                s2[row][0]=(a2[row]-mean3)/range3
                s2[row][1]=(m2[row]-mean4)/range4
                s3[row][0]=(a3[row]-mean5)/range5
                s3[row][1]=(m3[row]-mean6)/range6
                s4[row][0]=(a4[row]-mean7)/range7
                s4[row][1]=(m4[row]-mean8)/range8
            A = np.array(s1)
            B = np.array(s2)
            C = np.array(s3)
            D = np.array(s4)

            k1 = KMeans(n_clusters = 4, random_state=42)
            k1.fit(A)
            c1 = k1.cluster_centers_
            l1 = k1.labels_

            k2 = KMeans(n_clusters = 4, random_state=42)
            k2.fit(B)
            c2 = k2.cluster_centers_
            l2 = k2.labels_

            k3 = KMeans(n_clusters = 4, random_state=42)
            k3.fit(C)
            c3 = k3.cluster_centers_
            l3 = k3.labels_

            k4 = KMeans(n_clusters = 4, random_state=42)
            k4.fit(D)
            c4 = k4.cluster_centers_
            l4 = k4.labels_

            fig = plt.figure()
            colors = 10*['r.','g.','b.','c.','k.','y.','m.']
            plot1 = fig.add_subplot(2,2,1)
            for i in range(len(A)):
                print("coordinate:",A[i],"label:",l1[i])
                plot1.plot(A[i][0], A[i][1], colors[l1[i]], markersize = 10)
    
            plot1.scatter(c1[:, 0], c1[:, 1], marker = 'x', s = 150, linewidths = 5, zorder = 10)
            plot1.set_xlabel("Attendance")
            plot1.set_ylabel("Marks")
            plot1.set_title("Subject_1")

            plot2 = fig.add_subplot(2,2,2)
            for i in range(len(B)):
                print("coordinate:",B[i],"label:",l2[i])
                plot2.plot(B[i][0], B[i][1], colors[l2[i]], markersize = 10)
    
            plot2.scatter(c2[:, 0], c2[:, 1], marker = 'x', s = 150, linewidths = 5, zorder = 10)
            plot2.set_xlabel("Attendance")
            plot2.set_ylabel("Marks")
            plot2.set_title("Subject_2")

            plot3 = fig.add_subplot(2,2,3)
            for i in range(len(C)):
                print("coordinate:",C[i],"label:",l3[i])
                plot3.plot(C[i][0], C[i][1], colors[l3[i]], markersize = 10)
    
            plot3.scatter(c3[:, 0], c3[:, 1], marker = 'x', s = 150, linewidths = 5, zorder = 10)
            plot3.set_xlabel("Attendance")
            plot3.set_ylabel("Marks")
            plot3.set_title("Subject_3")

            plot4 = fig.add_subplot(2,2,4)
            for i in range(len(D)):
                print("coordinate:",D[i],"label:",l4[i])
                plot4.plot(D[i][0], D[i][1], colors[l4[i]], markersize = 10)
    
            plot4.scatter(c4[:, 0], c4[:, 1], marker = 'x', s = 150, linewidths = 5, zorder = 10)
            plot4.set_xlabel("Attendance")
            plot4.set_ylabel("Marks")
            plot4.set_title("Subject_4")
        
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            plt.show()

        #correlation indices of each
            n = len(a1)
            am1 = []
            for i in range(len(a1)):
                am1.append(float(a1[i]*m1[i]))
            r1 = (sum(am1) - n*st.mean(a1)*st.mean(m1))/(n*st.stdev(a1)*st.stdev(m1))
            print("Correlation index: ",r1)
            am2 = []
            for i in range(len(a1)):
                am2.append(float(a2[i]*m2[i]))
            r2 = (sum(am2) - n*st.mean(a2)*st.mean(m2))/(n*st.stdev(a2)*st.stdev(m2))
            print("Correlation index: ",r2)
            am3 = []
            for i in range(len(a1)):
                am3.append(float(a3[i]*m3[i]))
            r3 = (sum(am3) - n*st.mean(a3)*st.mean(m3))/(n*st.stdev(a3)*st.stdev(m3))
            print("Correlation index: ",r3)
            am4 = []
            for i in range(len(a1)):
                am4.append(float(a4[i]*m4[i]))
            r4 = (sum(am4) - n*st.mean(a4)*st.mean(m4))/(n*st.stdev(a4)*st.stdev(m4))
            print("Correlation index: ",r4)
            co = [r1,r2,r3,r4]
            N = 4
            width = 0.35
            ind = np.arange(N)
    
            text1 = "Correlation index for subject_1"+" = "+str(r1)
            text2 = "Correlation index for subject_2"+" = "+str(r2)
            text3 = "Correlation index for subject_3"+" = "+str(r3)
            text4 = "Correlation index for subject_4"+" = "+str(r4)
            txtfile = open('Example2.txt','a')
            txtfile.write(text1)
            txtfile.write(text2)
            txtfile.write(text3)
            txtfile.write(text4)
            txtfile.close()
        
            plt.bar(ind,co,width,color='r',bottom=0)
            plt.ylim(0,1)
            plt.title("Correlation Graph for all subjects")
            plt.xlabel("Subjects")
            plt.ylabel("Correlation index")
        
            plt.show()

        if l==10:
            with open(fname,'r') as csvfile:
                readCSV = csv.reader(csvfile,delimiter=',')
            
                a1 = []
                m1 = []
                a2 = []
                m2 = []
                a3 = []
                m3 = []
                a4 = []
                m4 = []
                a5 = []
                m5 = []
                s1 = []
                s2 = []
                s3 = []
                s4 = []
                s5 = []
                for row in readCSV:
                    a = row[0]
                    b = row[1]
                    c = row[2]
                    d = row[3]
                    e = row[4]
                    f = row[5]
                    g = row[6]
                    h = row[7]
                    i = row[8]
                    j = row[9]
                    a1.append(float(a))
                    m1.append(float(b))
                    a2.append(float(c))
                    m2.append(float(d))
                    a3.append(float(e))
                    m3.append(float(f))
                    a4.append(float(g))
                    m4.append(float(h))
                    a5.append(float(i))
                    m5.append(float(j))
                for row in range(len(a1)):
                    p = [a1[row],m1[row]]
                    q = [a2[row],m2[row]]
                    r = [a3[row],m3[row]]
                    s = [a4[row],m4[row]]
                    t = [a5[row],m5[row]]
                    s1.append(p)
                    s2.append(q)
                    s3.append(r)
                    s4.append(s)
                    s5.append(t)
            mean1 = st.mean(a1)
            mean2 = st.mean(m1)
            mean3 = st.mean(a2)
            mean4 = st.mean(m2)
            mean5 = st.mean(a3)
            mean6 = st.mean(m3)
            mean7 = st.mean(a4)
            mean8 = st.mean(m4)
            mean9 = st.mean(a5)
            mean10 = st.mean(m5)
            range1 = max(a1)-min(a1)
            range2 = max(m1)-min(m1)
            range3 = max(a2)-min(a2)
            range4 = max(m2)-min(m2)
            range5 = max(a3)-min(a3)
            range6 = max(m3)-min(m3)
            range7 = max(a4)-min(a4)
            range8 = max(m4)-min(m4)
            range9 = max(a5)-min(a5)
            range10 = max(m5)-min(m5)
            #normalizing
            for row in range(len(a1)):
                s1[row][0]=(a1[row]-mean1)/range1
                s1[row][1]=(m1[row]-mean2)/range2
                s2[row][0]=(a2[row]-mean3)/range3
                s2[row][1]=(m2[row]-mean4)/range4
                s3[row][0]=(a3[row]-mean5)/range5
                s3[row][1]=(m3[row]-mean6)/range6
                s4[row][0]=(a4[row]-mean7)/range7
                s4[row][1]=(m4[row]-mean8)/range8
                s5[row][0]=(a5[row]-mean9)/range9
                s5[row][1]=(m5[row]-mean10)/range10
            
            A = np.array(s1)
            B = np.array(s2)
            C = np.array(s3)
            D = np.array(s4)
            E = np.array(s5)

            k1 = KMeans(n_clusters = 4, random_state=42)
            k1.fit(A)
            c1 = k1.cluster_centers_
            l1 = k1.labels_

            k2 = KMeans(n_clusters = 4, random_state=42)
            k2.fit(B)
            c2 = k2.cluster_centers_
            l2 = k2.labels_

            k3 = KMeans(n_clusters = 4, random_state=42)
            k3.fit(C)
            c3 = k3.cluster_centers_
            l3 = k3.labels_

            k4 = KMeans(n_clusters = 4, random_state=42)
            k4.fit(D)
            c4 = k4.cluster_centers_
            l4 = k4.labels_
        
            k5 = KMeans(n_clusters = 4, random_state=42)
            k5.fit(E)
            c5 = k5.cluster_centers_
            l5 = k5.labels_

            fig = plt.figure()
            colors = 10*['r.','g.','b.','c.','k.','y.','m.']
            plot1 = fig.add_subplot(2,3,1)
            for i in range(len(A)):
                print("coordinate:",A[i],"label:",l1[i])
                plot1.plot(A[i][0], A[i][1], colors[l1[i]], markersize = 10)
    
            plot1.scatter(c1[:, 0], c1[:, 1], marker = 'x', s = 150, linewidths = 5, zorder = 10)
            plot1.set_xlabel("Attendance")
            plot1.set_ylabel("Marks")
            plot1.set_title("Subject_1")

            plot2 = fig.add_subplot(2,3,2)
            for i in range(len(B)):
                print("coordinate:",B[i],"label:",l2[i])
                plot2.plot(B[i][0], B[i][1], colors[l2[i]], markersize = 10)
    
            plot2.scatter(c2[:, 0], c2[:, 1], marker = 'x', s = 150, linewidths = 5, zorder = 10)
            plot2.set_xlabel("Attendance")
            plot2.set_ylabel("Marks")
            plot2.set_title("Subject_2")

            plot3 = fig.add_subplot(2,3,3)
            for i in range(len(C)):
                print("coordinate:",C[i],"label:",l3[i])
                plot3.plot(C[i][0], C[i][1], colors[l3[i]], markersize = 10)
    
            plot3.scatter(c3[:, 0], c3[:, 1], marker = 'x', s = 150, linewidths = 5, zorder = 10)
            plot3.set_xlabel("Attendance")
            plot3.set_ylabel("Marks")
            plot3.set_title("Subject_3")

            plot4 = fig.add_subplot(2,3,4)
            for i in range(len(D)):
                print("coordinate:",D[i],"label:",l4[i])
                plot4.plot(D[i][0], D[i][1], colors[l4[i]], markersize = 10)

            plot4.scatter(c4[:, 0], c4[:, 1], marker = 'x', s = 150, linewidths = 5, zorder = 10)
            plot4.set_xlabel("Attendance")
            plot4.set_ylabel("Marks")
            plot4.set_title("Subject_4")

            plot5 = fig.add_subplot(2,3,5)
            for i in range(len(E)):
                print("coordinate:",E[i],"label:",l5[i])
                plot5.plot(E[i][0], E[i][1], colors[l5[i]], markersize = 10)

            plot5.scatter(c5[:, 0], c5[:, 1], marker = 'x', s = 150, linewidths = 5, zorder = 10)
            plot5.set_xlabel("Attendance")
            plot5.set_ylabel("Marks")
            plot5.set_title("Subject_5")        
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            plt.show()


            #correlation indices of each
            n = len(a1)
            am1 = []
            for i in range(len(a1)):
                am1.append(float(a1[i]*m1[i]))
            r1 = (sum(am1) - n*st.mean(a1)*st.mean(m1))/(n*st.stdev(a1)*st.stdev(m1))
            print("Correlation index: ",r1)
            am2 = []
            for i in range(len(a1)):
                am2.append(float(a2[i]*m2[i]))
            r2 = (sum(am2) - n*st.mean(a2)*st.mean(m2))/(n*st.stdev(a2)*st.stdev(m2))
            print("Correlation index: ",r2)
            am3 = []
            for i in range(len(a1)):
                am3.append(float(a3[i]*m3[i]))
            r3 = (sum(am3) - n*st.mean(a3)*st.mean(m3))/(n*st.stdev(a3)*st.stdev(m3))
            print("Correlation index: ",r3)
            am4 = []
            for i in range(len(a1)):
                am4.append(float(a4[i]*m4[i]))
            r4 = (sum(am4) - n*st.mean(a4)*st.mean(m4))/(n*st.stdev(a4)*st.stdev(m4))
            print("Correlation index: ",r4)
            am5 = []
            for i in range(len(a1)):
                am5.append(float(a5[i]*m5[i]))
            r5 = (sum(am5) - n*st.mean(a5)*st.mean(m5))/(n*st.stdev(a5)*st.stdev(m5))
            print("Correlation index: ",r5)
            co = [r1,r2,r3,r4,r5]
            N = 5
            ind = np.arange(N)
            width = 0.35
            
            #cor = np.array(co)
    
            text1 = "Correlation index for subject_1"+" = "+str(r1)
            text2 = "Correlation index for subject_2"+" = "+str(r2)
            text3 = "Correlation index for subject_3"+" = "+str(r3)
            text4 = "Correlation index for subject_4"+" = "+str(r4)
            text5 = "Correlation index for subject_5"+" = "+str(r5)
            txtfile = open('Example2.txt','a')
            txtfile.write(text1)
            txtfile.write(text2)
            txtfile.write(text3)
            txtfile.write(text4)
            txtfile.write(text5)
            txtfile.close()
        
            plt.bar(ind,co,width,color='r',bottom=0)
            plt.ylim(0,1)
            plt.title("Correlation Graph for all subjects") 
            plt.show()

            
        if l==12:
            with open(fname,'r') as csvfile:
                readCSV = csv.reader(csvfile,delimiter=',')
            
                a1 = []
                m1 = []
                a2 = []
                m2 = []
                a3 = []
                m3 = []
                a4 = []
                m4 = []
                a5 = []
                m5 = []
                a6 = []
                m6 = []
                s1 = []
                s2 = []
                s3 = []
                s4 = []
                s5 = []
                s6 = []
                for row in readCSV:
                    a = row[0]
                    b = row[1]
                    c = row[2]
                    d = row[3]
                    e = row[4]
                    f = row[5]
                    g = row[6]
                    h = row[7]
                    i = row[8]
                    j = row[9]
                    k = row[10]
                    l = row[11]
                    a1.append(float(a))
                    m1.append(float(b))
                    a2.append(float(c))
                    m2.append(float(d))
                    a3.append(float(e))
                    m3.append(float(f))
                    a4.append(float(g))
                    m4.append(float(h))
                    a5.append(float(i))
                    m5.append(float(j))
                    a6.append(float(k))
                    m6.append(float(l))
                for row in range(len(a1)):
                    p = [a1[row],m1[row]]
                    q = [a2[row],m2[row]]
                    r = [a3[row],m3[row]]
                    s = [a4[row],m4[row]]
                    t = [a5[row],m5[row]]
                    u = [a6[row],m6[row]]
                    s1.append(p)
                    s2.append(q)
                    s3.append(r)
                    s4.append(s)
                    s5.append(t)
                    s6.append(u)
            mean1 = st.mean(a1)
            mean2 = st.mean(m1)
            mean3 = st.mean(a2)
            mean4 = st.mean(m2)
            mean5 = st.mean(a3)
            mean6 = st.mean(m3)
            mean7 = st.mean(a4)
            mean8 = st.mean(m4)
            mean9 = st.mean(a5)
            mean10 = st.mean(m5)
            mean11 = st.mean(a6)
            mean12 = st.mean(m6)
            range1 = max(a1)-min(a1)
            range2 = max(m1)-min(m1)
            range3 = max(a2)-min(a2)
            range4 = max(m2)-min(m2)
            range5 = max(a3)-min(a3)
            range6 = max(m3)-min(m3)
            range7 = max(a4)-min(a4)
            range8 = max(m4)-min(m4)
            range9 = max(a5)-min(a5)
            range10 = max(m5)-min(m5)
            range11 = max(a6)-min(a6)
            range12 = max(m6)-min(m6)
            #normalizing
            for row in range(len(a1)):
                s1[row][0]=(a1[row]-mean1)/range1
                s1[row][1]=(m1[row]-mean2)/range2
                s2[row][0]=(a2[row]-mean3)/range3
                s2[row][1]=(m2[row]-mean4)/range4
                s3[row][0]=(a3[row]-mean5)/range5
                s3[row][1]=(m3[row]-mean6)/range6
                s4[row][0]=(a4[row]-mean7)/range7
                s4[row][1]=(m4[row]-mean8)/range8
                s5[row][0]=(a5[row]-mean9)/range9
                s5[row][1]=(m5[row]-mean10)/range10
                s6[row][0]=(a6[row]-mean11)/range11
                s6[row][1]=(m6[row]-mean12)/range12
                        
            A = np.array(s1)
            B = np.array(s2)
            C = np.array(s3)
            D = np.array(s4)
            E = np.array(s5)
            F = np.array(s6)

            k1 = KMeans(n_clusters = 4, random_state=42)
            k1.fit(A)
            c1 = k1.cluster_centers_
            l1 = k1.labels_

            k2 = KMeans(n_clusters = 4, random_state=42)
            k2.fit(B)
            c2 = k2.cluster_centers_
            l2 = k2.labels_

            k3 = KMeans(n_clusters = 4, random_state=42)
            k3.fit(C)
            c3 = k3.cluster_centers_
            l3 = k3.labels_

            k4 = KMeans(n_clusters = 4, random_state=42)
            k4.fit(D)
            c4 = k4.cluster_centers_
            l4 = k4.labels_
        
            k5 = KMeans(n_clusters = 4, random_state=42)
            k5.fit(E)
            c5 = k5.cluster_centers_
            l5 = k5.labels_
        
            k6 = KMeans(n_clusters = 4, random_state=42)
            k6.fit(F)
            c6 = k6.cluster_centers_
            l6 = k6.labels_

            fig = plt.figure()
            colors = 10*['r.','g.','b.','c.','k.','y.','m.']
            plot1 = fig.add_subplot(2,3,1)
            for i in range(len(A)):
                print("coordinate:",A[i],"label:",l1[i])
                plot1.plot(A[i][0], A[i][1], colors[l1[i]], markersize = 10)
    
            plot1.scatter(c1[:, 0], c1[:, 1], marker = 'x', s = 150, linewidths = 5, zorder = 10)
            plot1.set_xlabel("Attendance")
            plot1.set_ylabel("Marks")
            plot1.set_title("Subject_1")

            plot2 = fig.add_subplot(2,3,2)
            for i in range(len(B)):
                print("coordinate:",B[i],"label:",l2[i])
                plot2.plot(B[i][0], B[i][1], colors[l2[i]], markersize = 10)
    
            plot2.scatter(c2[:, 0], c2[:, 1], marker = 'x', s = 150, linewidths = 5, zorder = 10)
            plot2.set_xlabel("Attendance")
            plot2.set_ylabel("Marks")
            plot2.set_title("Subject_2")

            plot3 = fig.add_subplot(2,3,3)
            for i in range(len(C)):
                print("coordinate:",C[i],"label:",l3[i])
                plot3.plot(C[i][0], C[i][1], colors[l3[i]], markersize = 10)
    
            plot3.scatter(c3[:, 0], c3[:, 1], marker = 'x', s = 150, linewidths = 5, zorder = 10)
            plot3.set_xlabel("Attendance")
            plot3.set_ylabel("Marks")
            plot3.set_title("Subject_3")

            plot4 = fig.add_subplot(2,3,4)
            for i in range(len(D)):
                print("coordinate:",D[i],"label:",l4[i])
                plot4.plot(D[i][0], D[i][1], colors[l4[i]], markersize = 10)

            plot4.scatter(c4[:, 0], c4[:, 1], marker = 'x', s = 150, linewidths = 5, zorder = 10)
            plot4.set_xlabel("Attendance")
            plot4.set_ylabel("Marks")
            plot4.set_title("Subject_4")
    
            plot5 = fig.add_subplot(2,3,5)
            for i in range(len(E)):
                print("coordinate:",E[i],"label:",l5[i])
                plot4.plot(E[i][0], E[i][1], colors[l5[i]], markersize = 10)

            plot5.scatter(c5[:, 0], c5[:, 1], marker = 'x', s = 150, linewidths = 5, zorder = 10)
            plot5.set_xlabel("Attendance")
            plot5.set_ylabel("Marks")
            plot5.set_title("Subject_5")

        
            plot6 = fig.add_subplot(2,3,6)
            for i in range(len(F)):
                print("coordinate:",F[i],"label:",l6[i])
                plot4.plot(F[i][0], F[i][1], colors[l6[i]], markersize = 10)
    
            plot6.scatter(c6[:, 0], c6[:, 1], marker = 'x', s = 150, linewidths = 5, zorder = 10)
            plot6.set_xlabel("Attendance")
            plot6.set_ylabel("Marks")
            plot6.set_title("Subject_6") 
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            plt.show()


            #correlation indices of each
            n = len(a1)
            am1 = []
            for i in range(len(a1)):
                am1.append(float(a1[i]*m1[i]))
            r1 = (sum(am1) - n*st.mean(a1)*st.mean(m1))/(n*st.stdev(a1)*st.stdev(m1))
            print("Correlation index: ",r1)
            am2 = []
            for i in range(len(a1)):
                am2.append(float(a2[i]*m2[i]))
            r2 = (sum(am2) - n*st.mean(a2)*st.mean(m2))/(n*st.stdev(a2)*st.stdev(m2))
            print("Correlation index: ",r2)
            am3 = []
            for i in range(len(a1)):
                am3.append(float(a3[i]*m3[i]))
            r3 = (sum(am3) - n*st.mean(a3)*st.mean(m3))/(n*st.stdev(a3)*st.stdev(m3))
            print("Correlation index: ",r3)
            am4 = []
            for i in range(len(a1)):
                am4.append(float(a4[i]*m4[i]))
            r4 = (sum(am4) - n*st.mean(a4)*st.mean(m4))/(n*st.stdev(a4)*st.stdev(m4))
            print("Correlation index: ",r4)
            am5 = []
            for i in range(len(a1)):
                am5.append(float(a5[i]*m5[i]))
            r5 = (sum(am5) - n*st.mean(a5)*st.mean(m5))/(n*st.stdev(a5)*st.stdev(m5))
            print("Correlation index: ",r5)
            am6 = []
            for i in range(len(a1)):
                am6.append(float(a6[i]*m6[i]))
            r6 = (sum(am6) - n*st.mean(a6)*st.mean(m6))/(n*st.stdev(a6)*st.stdev(m6))
            print("Correlation index: ",r6)
            co = [r1,r2,r3,r4,r5,r6]
            N = 6
            ind = np.arange(N)
            width = 0.35
        
            text1 = "Correlation index for subject_1"+" = "+str(r1)
            text2 = "Correlation index for subject_2"+" = "+str(r2)
            text3 = "Correlation index for subject_3"+" = "+str(r3)
            text4 = "Correlation index for subject_4"+" = "+str(r4)
            text5 = "Correlation index for subject_5"+" = "+str(r5)
            text6 = "Correlation index for subject_6"+" = "+str(r6)
            txtfile = open('Example2.txt','a')
            txtfile.write(text1)
            txtfile.write(text2)
            txtfile.write(text3)
            txtfile.write(text4)
            txtfile.write(text5)
            txtfile.write(text6)
            txtfile.close()
            
            plt.bar(ind,co,width,color='r',bottom=0)
            plt.ylim(0,1)
            plt.title("Correlation Graph for all subjects") 
            plt.show()
            

            
    
    
    def report1(self):
        print("report1")
        
    def report2(self):
        print("report2")
    
    
        

class LoginWidget(QtGui.QWidget):

    def __init__(self,parent=None):
        QtGui.QWidget.__init__(self)
        super(LoginWidget,self).__init__(parent)
        self.setupUi(self)
        
    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(800, 600)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        MainWindow.setFont(font)
        MainWindow.setInputMethodHints(QtCore.Qt.ImhHiddenText)
        #MainWindow.setIconSize(QtCore.QSize(24, 24))
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.label = QtGui.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(220, 40, 391, 31))
        font = QtGui.QFont()
        font.setPointSize(15)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setFrameShape(QtGui.QFrame.NoFrame)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setWordWrap(False)
        self.label.setObjectName(_fromUtf8("label"))
        self.line = QtGui.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(17, 20, 761, 20))
        self.line.setFrameShape(QtGui.QFrame.HLine)
        self.line.setFrameShadow(QtGui.QFrame.Sunken)
        self.line.setObjectName(_fromUtf8("line"))
        self.line_2 = QtGui.QFrame(self.centralwidget)
        self.line_2.setGeometry(QtCore.QRect(20, 70, 761, 20))
        self.line_2.setFrameShape(QtGui.QFrame.HLine)
        self.line_2.setFrameShadow(QtGui.QFrame.Sunken)
        self.line_2.setObjectName(_fromUtf8("line_2"))
        self.formLayoutWidget = QtGui.QWidget(self.centralwidget)
        self.formLayoutWidget.setGeometry(QtCore.QRect(249, 160, 341, 271))
        self.formLayoutWidget.setObjectName(_fromUtf8("formLayoutWidget"))
        self.formLayout_2 = QtGui.QFormLayout(self.formLayoutWidget)
        self.formLayout_2.setObjectName(_fromUtf8("formLayout_2"))
        self.emailIDLabel = QtGui.QLabel(self.formLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.emailIDLabel.setFont(font)
        self.emailIDLabel.setObjectName(_fromUtf8("emailIDLabel"))
        self.formLayout_2.setWidget(0, QtGui.QFormLayout.LabelRole, self.emailIDLabel)
        self.emailIDLineEdit = QtGui.QLineEdit(self.formLayoutWidget)
        self.emailIDLineEdit.setObjectName(_fromUtf8("emailIDLineEdit"))
        self.formLayout_2.setWidget(0, QtGui.QFormLayout.FieldRole, self.emailIDLineEdit)
        spacerItem = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.formLayout_2.setItem(1, QtGui.QFormLayout.SpanningRole, spacerItem)
        self.passwordLabel = QtGui.QLabel(self.formLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.passwordLabel.setFont(font)
        self.passwordLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.passwordLabel.setObjectName(_fromUtf8("passwordLabel"))
        self.formLayout_2.setWidget(2, QtGui.QFormLayout.LabelRole, self.passwordLabel)
        self.passwordLineEdit = QtGui.QLineEdit(self.formLayoutWidget)
        self.passwordLineEdit.setEchoMode(QtGui.QLineEdit.Password)
        self.passwordLineEdit.setInputMethodHints(QtCore.Qt.ImhHiddenText)
        self.passwordLineEdit.setObjectName(_fromUtf8("passwordLineEdit"))
        self.formLayout_2.setWidget(2, QtGui.QFormLayout.FieldRole, self.passwordLineEdit)
        spacerItem1 = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.formLayout_2.setItem(3, QtGui.QFormLayout.SpanningRole, spacerItem1)
        self.pushButton = QtGui.QPushButton(self.formLayoutWidget)
        self.pushButton.setObjectName(_fromUtf8("pushButton"))
        self.formLayout_2.setWidget(4, QtGui.QFormLayout.SpanningRole, self.pushButton)
        spacerItem2 = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.formLayout_2.setItem(5, QtGui.QFormLayout.SpanningRole, spacerItem2)
        self.label_2 = QtGui.QLabel(self.formLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setUnderline(True)
        self.label_2.setFont(font)
        self.label_2.setOpenExternalLinks(True)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.formLayout_2.setWidget(6, QtGui.QFormLayout.FieldRole, self.label_2)
        #MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        self.menuFile = QtGui.QMenu(self.menubar)
        self.menuFile.setObjectName(_fromUtf8("menuFile"))
        #MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        #MainWindow.setStatusBar(self.statusbar)
        self.actionEXit = QtGui.QAction(MainWindow)
        self.actionEXit.setObjectName(_fromUtf8("actionEXit"))
        self.menuFile.addAction(self.actionEXit)
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "Py-Analyse - Login", None))
        self.label.setText(_translate("MainWindow", "Welcome to Py_analyse", None))
        self.emailIDLabel.setText(_translate("MainWindow", "       Email ID :      ", None))
        self.passwordLabel.setText(_translate("MainWindow", "      Password : ", None))
        self.pushButton.setText(_translate("MainWindow", "Login", None))
        self.label_2.setText(_translate("MainWindow", "Forgot Password ?", None))
        self.menuFile.setTitle(_translate("MainWindow", "File", None))
        self.actionEXit.setText(_translate("MainWindow", "Exit", None))
        #self.pushButton.clicked.connect(self.login_)

    def login_(self):
        e = str(self.emailIDLineEdit.text())
        p = str(self.passwordLineEdit.text())
        if e =="nachimak28@gmail.com" and p=="123":
            print("Login successful")
        else:
            print("Enter proper credentials")
            msgbox(self)

    def msgbox(self):
        msg = QtGui.QMessageBox()
        msg.setIcon(QtGui.QMessageBox.Critical)

        msg.setText("Incorrect Credentials")
        msg.setInformativeText("Please enter correct credentials.")
        msg.setWindowTitle("Incorrect Crendentials")
        #msg.setDetailedText("The details are as follows:")
        msg.setStandardButtons(QtGui.QMessageBox.Ok | QtGui.QMessageBox.Cancel)
        retval = msg.exec_()
            
class SecondScreen(QtGui.QWidget):

    def __init__(self,parent=None):
        QtGui.QWidget.__init__(self)
        super(SecondScreen,self).__init__(parent)
        self.setupUi(self)
        
    
    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(800, 600)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        MainWindow.setFont(font)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.label = QtGui.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(340, 10, 141, 61))
        font = QtGui.QFont()
        font.setPointSize(18)
        font.setBold(True)
        font.setItalic(True)
        font.setUnderline(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName(_fromUtf8("label"))
        self.label_2 = QtGui.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(260, 70, 311, 20))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.formLayoutWidget = QtGui.QWidget(self.centralwidget)
        self.formLayoutWidget.setGeometry(QtCore.QRect(20, 110, 761, 451))
        self.formLayoutWidget.setObjectName(_fromUtf8("formLayoutWidget"))
        self.formLayout = QtGui.QFormLayout(self.formLayoutWidget)
        self.formLayout.setObjectName(_fromUtf8("formLayout"))
        self.selectDepartmentLabel = QtGui.QLabel(self.formLayoutWidget)
        self.selectDepartmentLabel.setObjectName(_fromUtf8("selectDepartmentLabel"))
        self.formLayout.setWidget(0, QtGui.QFormLayout.LabelRole, self.selectDepartmentLabel)
        self.selectDepartmentComboBox = QtGui.QComboBox(self.formLayoutWidget)
        self.selectDepartmentComboBox.setObjectName(_fromUtf8("selectDepartmentComboBox"))
        self.selectDepartmentComboBox.addItem(_fromUtf8(""))
        self.selectDepartmentComboBox.addItem(_fromUtf8(""))
        self.selectDepartmentComboBox.addItem(_fromUtf8(""))
        self.formLayout.setWidget(0, QtGui.QFormLayout.FieldRole, self.selectDepartmentComboBox)
        spacerItem = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.formLayout.setItem(1, QtGui.QFormLayout.SpanningRole, spacerItem)
        self.selectYearLabel = QtGui.QLabel(self.formLayoutWidget)
        self.selectYearLabel.setObjectName(_fromUtf8("selectYearLabel"))
        self.formLayout.setWidget(2, QtGui.QFormLayout.LabelRole, self.selectYearLabel)
        self.selectYearComboBox = QtGui.QComboBox(self.formLayoutWidget)
        self.selectYearComboBox.setObjectName(_fromUtf8("selectYearComboBox"))
        self.selectYearComboBox.addItem(_fromUtf8(""))
        self.selectYearComboBox.addItem(_fromUtf8(""))
        self.selectYearComboBox.addItem(_fromUtf8(""))
        self.selectYearComboBox.addItem(_fromUtf8(""))
        self.formLayout.setWidget(2, QtGui.QFormLayout.FieldRole, self.selectYearComboBox)
        spacerItem1 = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.formLayout.setItem(3, QtGui.QFormLayout.SpanningRole, spacerItem1)
        self.chooseSemesterLabel = QtGui.QLabel(self.formLayoutWidget)
        self.chooseSemesterLabel.setObjectName(_fromUtf8("chooseSemesterLabel"))
        self.formLayout.setWidget(4, QtGui.QFormLayout.LabelRole, self.chooseSemesterLabel)
        self.chooseSemesterWidget = QtGui.QWidget(self.formLayoutWidget)
        self.chooseSemesterWidget.setObjectName(_fromUtf8("chooseSemesterWidget"))
        self.radioButton = QtGui.QRadioButton(self.chooseSemesterWidget)
        self.radioButton.setGeometry(QtCore.QRect(110, 0, 82, 17))
        self.radioButton.setObjectName(_fromUtf8("radioButton"))
        self.radioButton_2 = QtGui.QRadioButton(self.chooseSemesterWidget)
        self.radioButton_2.setGeometry(QtCore.QRect(290, 0, 82, 17))
        self.radioButton_2.setObjectName(_fromUtf8("radioButton_2"))
        self.formLayout.setWidget(4, QtGui.QFormLayout.FieldRole, self.chooseSemesterWidget)
        spacerItem2 = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.formLayout.setItem(5, QtGui.QFormLayout.SpanningRole, spacerItem2)
        self.chooseDivisionWidget = QtGui.QWidget(self.formLayoutWidget)
        self.chooseDivisionWidget.setObjectName(_fromUtf8("chooseDivisionWidget"))
        self.radioButton_3 = QtGui.QRadioButton(self.chooseDivisionWidget)
        self.radioButton_3.setGeometry(QtCore.QRect(110, 0, 82, 17))
        self.radioButton_3.setObjectName(_fromUtf8("radioButton_3"))
        self.radioButton_4 = QtGui.QRadioButton(self.chooseDivisionWidget)
        self.radioButton_4.setGeometry(QtCore.QRect(290, 0, 82, 17))
        self.radioButton_4.setObjectName(_fromUtf8("radioButton_4"))
        self.formLayout.setWidget(6, QtGui.QFormLayout.FieldRole, self.chooseDivisionWidget)
        spacerItem3 = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.formLayout.setItem(7, QtGui.QFormLayout.SpanningRole, spacerItem3)
        self.chooseDurationLabel = QtGui.QLabel(self.formLayoutWidget)
        self.chooseDurationLabel.setObjectName(_fromUtf8("chooseDurationLabel"))
        self.formLayout.setWidget(8, QtGui.QFormLayout.LabelRole, self.chooseDurationLabel)
        self.chooseDurationWidget = QtGui.QWidget(self.formLayoutWidget)
        self.chooseDurationWidget.setObjectName(_fromUtf8("chooseDurationWidget"))
        self.radioButton_5 = QtGui.QRadioButton(self.chooseDurationWidget)
        self.radioButton_5.setGeometry(QtCore.QRect(110, 0, 82, 17))
        self.radioButton_5.setObjectName(_fromUtf8("radioButton_5"))
        self.radioButton_6 = QtGui.QRadioButton(self.chooseDurationWidget)
        self.radioButton_6.setGeometry(QtCore.QRect(290, 0, 82, 17))
        self.radioButton_6.setObjectName(_fromUtf8("radioButton_6"))
        self.radioButton_7 = QtGui.QRadioButton(self.chooseDurationWidget)
        self.radioButton_7.setGeometry(QtCore.QRect(470, 0, 82, 17))
        self.radioButton_7.setObjectName(_fromUtf8("radioButton_7"))
        self.formLayout.setWidget(8, QtGui.QFormLayout.FieldRole, self.chooseDurationWidget)
        spacerItem4 = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.formLayout.setItem(9, QtGui.QFormLayout.SpanningRole, spacerItem4)
        self.pushButton = QtGui.QPushButton(self.formLayoutWidget)
        self.pushButton.setObjectName(_fromUtf8("pushButton"))
        self.formLayout.setWidget(10, QtGui.QFormLayout.FieldRole, self.pushButton)
        self.chooseDivisionLabel = QtGui.QLabel(self.formLayoutWidget)
        self.chooseDivisionLabel.setObjectName(_fromUtf8("chooseDivisionLabel"))
        self.formLayout.setWidget(6, QtGui.QFormLayout.LabelRole, self.chooseDivisionLabel)
        #MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 23))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        self.menuFile = QtGui.QMenu(self.menubar)
        self.menuFile.setObjectName(_fromUtf8("menuFile"))
        #MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        #MainWindow.setStatusBar(self.statusbar)
        self.actionExit = QtGui.QAction(MainWindow)
        self.actionExit.setObjectName(_fromUtf8("actionExit"))
        self.menuFile.addAction(self.actionExit)
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))
        self.label.setText(_translate("MainWindow", "Py- Analyse", None))
        self.label_2.setText(_translate("MainWindow", "Choose from the following options.", None))
        self.selectDepartmentLabel.setText(_translate("MainWindow", "             Select Department :     ", None))
        self.selectDepartmentComboBox.setItemText(0, _translate("MainWindow", "INFT", None))
        self.selectDepartmentComboBox.setItemText(1, _translate("MainWindow", "CMPN", None))
        self.selectDepartmentComboBox.setItemText(2, _translate("MainWindow", "EXTC", None))
        self.selectYearLabel.setText(_translate("MainWindow", "                         Select year :      ", None))
        self.selectYearComboBox.setItemText(0, _translate("MainWindow", "FE", None))
        self.selectYearComboBox.setItemText(1, _translate("MainWindow", "SE", None))
        self.selectYearComboBox.setItemText(2, _translate("MainWindow", "TE", None))
        self.selectYearComboBox.setItemText(3, _translate("MainWindow", "BE", None))
        self.chooseSemesterLabel.setText(_translate("MainWindow", "               Choose Semester :     ", None))
        self.radioButton.setText(_translate("MainWindow", "Even", None))
        self.radioButton_2.setText(_translate("MainWindow", "Odd", None))
        self.radioButton_3.setText(_translate("MainWindow", "A", None))
        self.radioButton_4.setText(_translate("MainWindow", "B", None))
        self.chooseDurationLabel.setText(_translate("MainWindow", "                 Choose Duration :      ", None))
        self.radioButton_5.setText(_translate("MainWindow", "SEM", None))
        self.radioButton_6.setText(_translate("MainWindow", "IAT1", None))
        self.radioButton_7.setText(_translate("MainWindow", "IAT2", None))
        self.pushButton.setText(_translate("MainWindow", "Next >>", None))
        self.chooseDivisionLabel.setText(_translate("MainWindow", "                  Choose Division :       ", None))
        self.menuFile.setTitle(_translate("MainWindow", "File", None))
        self.actionExit.setText(_translate("MainWindow", "Exit", None))
        
        
                #self.radioButton.toggled.connnect(lambda:self.btnstate0(self.radioButton))
        #self.radioButton_2.toggled.connnect(lambda:self.btnstate0(self.radioButton_2))
        #self.radioButton_3.toggled.connnect(lambda:self.btnstate1(self.radioButton_3))
        #self.radioButton_4.toggled.connnect(lambda:self.btnstate1(self.radioButton_4))
        #self.radioButton_5.toggled.connnect(lambda:self.btnstate2(self.radioButton_5))
        #self.radioButton_6.toggled.connnect(lambda:self.btnstate2(self.radioButton_6))
        #self.radioButton_7.toggled.connnect(lambda:self.btnstate2(self.radioButton_7))
        self.pushButton.clicked.connect(self.status)

    def status(self):
        if self.radioButton.isChecked()==True:
            print("Even is selected")
            self.s = "Even"
        if self.radioButton_2.isChecked()==True:
            print("Odd is selected")
            self.s = "Odd"
        if self.radioButton_3.isChecked()==True:
            print("A is selected")
            self.d = "A"
        if self.radioButton_4.isChecked()==True:
            print("B is selected")
            self.d = "B"
        if self.radioButton_5.isChecked()==True:
            print("SEM is selected")
            self.t = "SEM"
        if self.radioButton_6.isChecked()==True:
            print("IAT1 is selected")
            self.t = "IAT1"        
        if self.radioButton_7.isChecked()==True:
            print("IAT2 is selected")
            self.t = "IAT2"
        self.dept = self.selectDepartmentComboBox.currentText()
        self.yr = self.selectYearComboBox.currentText()
        print(self.yr,"-",self.dept,"-",self.d," ",self.s," sem",self.t," data has been selected for analysis")
        

class AnalyseFile(QtGui.QWidget):


    def __init__(self,parent=None):
        QtGui.QWidget.__init__(self)
        super(AnalyseFile,self).__init__(parent)
        self.setupUi(self)
        
  
    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(800, 600)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        MainWindow.setFont(font)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.label = QtGui.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(340, 20, 131, 41))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setItalic(True)
        font.setUnderline(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName(_fromUtf8("label"))
        self.line = QtGui.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(20, 110, 761, 16))
        self.line.setFrameShape(QtGui.QFrame.HLine)
        self.line.setFrameShadow(QtGui.QFrame.Sunken)
        self.line.setObjectName(_fromUtf8("line"))
        self.line_2 = QtGui.QFrame(self.centralwidget)
        self.line_2.setGeometry(QtCore.QRect(20, 330, 751, 16))
        self.line_2.setFrameShape(QtGui.QFrame.HLine)
        self.line_2.setFrameShadow(QtGui.QFrame.Sunken)
        self.line_2.setObjectName(_fromUtf8("line_2"))
        self.formLayoutWidget = QtGui.QWidget(self.centralwidget)
        self.formLayoutWidget.setGeometry(QtCore.QRect(30, 140, 741, 181))
        self.formLayoutWidget.setObjectName(_fromUtf8("formLayoutWidget"))
        self.formLayout_2 = QtGui.QFormLayout(self.formLayoutWidget)
        self.formLayout_2.setObjectName(_fromUtf8("formLayout_2"))
        self.uploadSingleSubjectFileForAnalysisLabel = QtGui.QLabel(self.formLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.uploadSingleSubjectFileForAnalysisLabel.setFont(font)
        self.uploadSingleSubjectFileForAnalysisLabel.setObjectName(_fromUtf8("uploadSingleSubjectFileForAnalysisLabel"))
        self.formLayout_2.setWidget(0, QtGui.QFormLayout.LabelRole, self.uploadSingleSubjectFileForAnalysisLabel)
        self.uploadSingleSubjectFileForAnalysisWidget = QtGui.QWidget(self.formLayoutWidget)
        self.uploadSingleSubjectFileForAnalysisWidget.setObjectName(_fromUtf8("uploadSingleSubjectFileForAnalysisWidget"))
        self.pushButton = QtGui.QPushButton(self.uploadSingleSubjectFileForAnalysisWidget)
        self.pushButton.setGeometry(QtCore.QRect(4, 0, 411, 21))
        self.pushButton.setObjectName(_fromUtf8("pushButton"))
        self.formLayout_2.setWidget(0, QtGui.QFormLayout.FieldRole, self.uploadSingleSubjectFileForAnalysisWidget)
        spacerItem = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.formLayout_2.setItem(1, QtGui.QFormLayout.SpanningRole, spacerItem)
        self.pushButton_2 = QtGui.QPushButton(self.formLayoutWidget)
        self.pushButton_2.setObjectName(_fromUtf8("pushButton_2"))
        self.formLayout_2.setWidget(2, QtGui.QFormLayout.FieldRole, self.pushButton_2)
        self.formLayoutWidget_2 = QtGui.QWidget(self.centralwidget)
        self.formLayoutWidget_2.setGeometry(QtCore.QRect(30, 360, 741, 111))
        self.formLayoutWidget_2.setObjectName(_fromUtf8("formLayoutWidget_2"))
        self.formLayout_3 = QtGui.QFormLayout(self.formLayoutWidget_2)
        self.formLayout_3.setObjectName(_fromUtf8("formLayout_3"))
        self.uploadSingleSubjectFileForAnalysisLabel_2 = QtGui.QLabel(self.formLayoutWidget_2)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.uploadSingleSubjectFileForAnalysisLabel_2.setFont(font)
        self.uploadSingleSubjectFileForAnalysisLabel_2.setObjectName(_fromUtf8("uploadSingleSubjectFileForAnalysisLabel_2"))
        self.formLayout_3.setWidget(0, QtGui.QFormLayout.LabelRole, self.uploadSingleSubjectFileForAnalysisLabel_2)
        self.uploadSingleSubjectFileForAnalysisWidget_2 = QtGui.QWidget(self.formLayoutWidget_2)
        self.uploadSingleSubjectFileForAnalysisWidget_2.setObjectName(_fromUtf8("uploadSingleSubjectFileForAnalysisWidget_2"))
        self.pushButton_3 = QtGui.QPushButton(self.uploadSingleSubjectFileForAnalysisWidget_2)
        self.pushButton_3.setGeometry(QtCore.QRect(4, 0, 411, 21))
        self.pushButton_3.setObjectName(_fromUtf8("pushButton_3"))
        self.formLayout_3.setWidget(0, QtGui.QFormLayout.FieldRole, self.uploadSingleSubjectFileForAnalysisWidget_2)
        spacerItem1 = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.formLayout_3.setItem(1, QtGui.QFormLayout.SpanningRole, spacerItem1)
        self.pushButton_4 = QtGui.QPushButton(self.formLayoutWidget_2)
        self.pushButton_4.setObjectName(_fromUtf8("pushButton_4"))
        self.formLayout_3.setWidget(2, QtGui.QFormLayout.FieldRole, self.pushButton_4)
        self.line_3 = QtGui.QFrame(self.centralwidget)
        self.line_3.setGeometry(QtCore.QRect(0, 0, 751, 16))
        self.line_3.setFrameShape(QtGui.QFrame.HLine)
        self.line_3.setFrameShadow(QtGui.QFrame.Sunken)
        self.line_3.setObjectName(_fromUtf8("line_3"))
        self.line_4 = QtGui.QFrame(self.centralwidget)
        self.line_4.setGeometry(QtCore.QRect(20, 480, 751, 16))
        self.line_4.setFrameShape(QtGui.QFrame.HLine)
        self.line_4.setFrameShadow(QtGui.QFrame.Sunken)
        self.line_4.setObjectName(_fromUtf8("line_4"))
        self.pushButton_5 = QtGui.QPushButton(self.centralwidget)
        self.pushButton_5.setGeometry(QtCore.QRect(360, 500, 141, 23))
        self.pushButton_5.setObjectName(_fromUtf8("pushButton_5"))
        self.pushButton_6 = QtGui.QPushButton(self.centralwidget)
        self.pushButton_6.setGeometry(QtCore.QRect(200, 500, 141, 23))
        self.pushButton_6.setObjectName(_fromUtf8("pushButton_6"))
        #MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 23))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        self.menuFile = QtGui.QMenu(self.menubar)
        self.menuFile.setObjectName(_fromUtf8("menuFile"))
        #MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        #MainWindow.setStatusBar(self.statusbar)
        self.actionExit = QtGui.QAction(MainWindow)
        self.actionExit.setObjectName(_fromUtf8("actionExit"))
        self.menuFile.addAction(self.actionExit)
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))
        self.label.setText(_translate("MainWindow", "Py- Analyse", None))
        self.uploadSingleSubjectFileForAnalysisLabel.setText(_translate("MainWindow", "Upload single subject file for analysis : ", None))
        self.pushButton.setText(_translate("MainWindow", "Analyse", None))
        self.pushButton_2.setText(_translate("MainWindow", "Generate Report", None))
        self.uploadSingleSubjectFileForAnalysisLabel_2.setText(_translate("MainWindow", "Upload all subject file for analysis :      ", None))
        self.pushButton_3.setText(_translate("MainWindow", "Analyse", None))
        self.pushButton_4.setText(_translate("MainWindow", "Generate Report", None))
        self.pushButton_5.setText(_translate("MainWindow", "Exit", None))
        self.pushButton_6.setText(_translate("MainWindow", "<< Previous", None))
        self.menuFile.setTitle(_translate("MainWindow", "File", None))
        self.actionExit.setText(_translate("MainWindow", "Exit", None))
        
        
if __name__ == '__main__':
    app = QtGui.QApplication([])
    window = MainWindow()
    window.show()
    window.resize(800,600)
    app.exec()
