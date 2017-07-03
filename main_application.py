from PyQt4 import QtCore, QtGui
import sys
import csv
import numpy as np
import statistics as st
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn.cluster import KMeans


class MainWindow(QtGui.QMainWindow):
    def __init__(self,parent=None):
        super(MainWindow, self).__init__(parent)
        self.central_widget = QtGui.QStackedWidget()
        self.setCentralWidget(self.central_widget)
        login_widget = LoginWidget(self)
        login_widget.b1.clicked.connect(self.postlogin)
        self.central_widget.addWidget(login_widget)
    def postlogin(self):
        
        logged_in_widget = LoggedWidget(self)
        self.central_widget.addWidget(logged_in_widget)
        self.central_widget.setCurrentWidget(logged_in_widget)
        logged_in_widget.next.clicked.connect(self.file)

    def file(self):
        upload_file = FileUpload(self)
        self.central_widget.addWidget(upload_file)
        self.central_widget.setCurrentWidget(upload_file)
        

class LoginWidget(QtGui.QWidget):
    def __init__(self, parent=None):
        super(LoginWidget,self).__init__(parent)

        self.l1 = QtGui.QLabel()
        self.l2 = QtGui.QLabel()
        self.l1.setText("Welcome to PyNalyse! , your data analysis pal ")
        #l2.setText("Forgot your password ?")
        self.fbox = QtGui.QFormLayout()
        #self.l1.setAlignment(Qt.AlignCenter)
        #l2.setAlignment(Qt.AlignCenter)
    
        self.vbox = QtGui.QVBoxLayout()
        self.vbox.addWidget(self.l1)
        self.vbox.addStretch()
        self.fbox.addRow(self.l1,self.vbox)
    
    
        self.l3 = QtGui.QLabel("Emain ID: ")
        self.eid = QtGui.QLineEdit()
        
        #eid.setInputMask('abc@xyz.com')

        self.l4 = QtGui.QLabel("Password: ")
        self.pwd = QtGui.QLineEdit()
        self.pwd.setEchoMode(QtGui.QLineEdit.Password)
        #   add2 = QLineEdit()
    
        self.fbox.addRow(self.l3,self.eid)
        self.vbox = QtGui.QVBoxLayout()

        self.vbox.addWidget(self.pwd)
        #   vbox.addWidget(add2)
        self.fbox.addRow(self.l4,self.vbox)

        self.b1 = QtGui.QPushButton("Login")
        self.vbox.addWidget(self.b1)
        self.vbox.addStretch()
    

        self.b2 = QtGui.QPushButton("Forgot password ?")
        self.vbox.addWidget(self.b2)
        self.vbox.addStretch()
    
        #fbox.addRow(b1,b2)
        self.l1.setOpenExternalLinks(True)
        #self.l2.linkActivated.connect(clicked)
        #l2.linkHovered.connect(clicked)
        #self.l2.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.b1.clicked.connect(self.login_)
        
        self.setLayout(self.fbox)
        self.setWindowTitle("PyNalyse Login")

    def login_(self):
        e = str(self.eid.text())
        p = str(self.pwd.text())
        if e=="nachimak28@gmail.com" and p=="123":
            print("Login successful!")
        else:
            print("Enter proper credentials")
        

class LoggedWidget(QtGui.QWidget):
    def __init__(self,parent=None):

        super(LoggedWidget, self).__init__(parent)
        QtGui.QWidget.__init__(self)
        self.label0 = QtGui.QLabel()
        #self.a = "nachimak28@gmail.com"
        self.label0.setText("Welcome nachimak28")
        #self.label0.setAlignment(Qt.AlignCenter)

        self.label = QtGui.QLabel()
        self.label.setText("Department : ")
        self.label1 = QtGui.QLabel()
        self.label1.setText("Year : ")
        
        
        self.label2 = QtGui.QLabel()
        self.label2.setText("Division : ")
        self.label3 = QtGui.QLabel()
        #self.label3.setText("Year : ")
        #self.label3 = QtGui.QLabel()
        self.label3.setText("Semester : ")
        self.label4 = QtGui.QLabel()
        self.label4.setText("Term : ")
        self.next = QtGui.QPushButton('Next>>',self)
        self.next.clicked.connect(self.next_)

        fbox = QtGui.QFormLayout()
        hbox0 = QtGui.QHBoxLayout()
        self.dep1 = QtGui.QRadioButton("COMPS")
        self.dep1.setChecked(True)
        self.dep1.toggled.connect(lambda:self.btnstate0(self.dep1))
        hbox0.addWidget(self.dep1)
        self.dep2 = QtGui.QRadioButton("INFT")
        self.dep2.setChecked(False)
        self.dep2.toggled.connect(lambda:self.btnstate0(self.dep2))
        hbox0.addWidget(self.dep2)
        self.dep3 = QtGui.QRadioButton("EXTC")
        self.dep3.setChecked(False)
        self.dep3.toggled.connect(lambda:self.btnstate0(self.dep3))
        hbox0.addWidget(self.dep3)
        fbox.addRow(self.label0)
        fbox.addRow(self.label,hbox0)
        
        hbox = QtGui.QHBoxLayout()
        
        self.y1 = QtGui.QRadioButton("FE")
        self.y1.setChecked(True)
        self.y1.toggled.connect(lambda:self.btnstate1(self.y1))
        hbox.addWidget(self.y1)
        self.y2 = QtGui.QRadioButton("SE")
        self.y2.setChecked(False)
        self.y2.toggled.connect(lambda:self.btnstate1(self.y2))
        hbox.addWidget(self.y2)
        self.y3 = QtGui.QRadioButton("TE")
        self.y3.setChecked(False)
        self.y3.toggled.connect(lambda:self.btnstate1(self.y3))
        hbox.addWidget(self.y3)
        self.y4 = QtGui.QRadioButton("BE")
        self.y4.setChecked(False)
        self.y4.toggled.connect(lambda:self.btnstate1(self.y4))
        hbox.addWidget(self.y4)
        
        fbox.addRow(self.label1,hbox)

        hbox1 = QtGui.QHBoxLayout()
        self.d1 = QtGui.QRadioButton("A")
        self.d1.setChecked(True)
        self.d1.toggled.connect(lambda:self.btnstate2(self.d1))
        hbox1.addWidget(self.d1)
        self.d2 = QtGui.QRadioButton("B")
        self.d2.setChecked(False)
        self.d2.toggled.connect(lambda:self.btnstate2(self.d2))
        hbox1.addWidget(self.d2)

        fbox.addRow(self.label2,hbox1)

        hbox2 = QtGui.QHBoxLayout()
        self.s1 = QtGui.QRadioButton("Even")
        self.s1.setChecked(True)
        self.s1.toggled.connect(lambda:self.btnstate3(self.s1))
        hbox2.addWidget(self.s1)
        self.s2 = QtGui.QRadioButton("Odd")
        self.s2.setChecked(False)
        self.s2.toggled.connect(lambda:self.btnstate3(self.s2))
        hbox2.addWidget(self.s2)

        fbox.addRow(self.label3,hbox2)

        hbox3 = QtGui.QHBoxLayout()
        self.t1 = QtGui.QRadioButton("IAT1")
        self.t1.setChecked(True)
        self.t1.toggled.connect(lambda:self.btnstate4(self.t1))
        hbox3.addWidget(self.t1)
        self.t2 = QtGui.QRadioButton("IAT2")
        self.t2.setChecked(False)
        self.t2.toggled.connect(lambda:self.btnstate4(self.t2))
        hbox3.addWidget(self.t2)
        self.t3 = QtGui.QRadioButton("SEM")
        self.t3.setChecked(False)
        self.t3.toggled.connect(lambda:self.btnstate4(self.t3))
        hbox3.addWidget(self.t3)

        fbox.addRow(self.label4,hbox3)

        fbox.addRow(self.next)

        self.setLayout(fbox)
        self.setWindowTitle("PyNalyse Select options")

    def btnstate0(self,b):
        #self.dep = "ABCD"
        if b.text() == "COMPS":
            if b.isChecked() == True:
                print(b.text()+" is selected")
                self.dep = str(b.text())
        if b.text() == "INFT":
            if b.isChecked() == True:
                print(b.text()+" is selected")
                self.dep = str(b.text())
        if b.text() == "EXTC":
            if b.isChecked() == True:
                print(b.text()+" is selected")
                self.dep = str(b.text())
    
    def btnstate1(self,b):
        #self.y = "E"
        if b.text() == "FE":
            if b.isChecked() == True:
                print(b.text()+" is selected")
                self.y = str(b.text())
        if b.text() == "SE":
            if b.isChecked() == True:
                print(b.text()+" is selected")
                self.y = str(b.text())
        if b.text() == "TE":
            if b.isChecked() == True:
                print(b.text()+" is selected")
                self.y = str(b.text())
        if b.text() == "BE":
            if b.isChecked() == True:
                print(b.text()+" is selected")
                self.y = str(b.text())
    
    def btnstate2(self,b):
        #self.d = "O"
        if b.text() == "A":
            if b.isChecked() == True:
                print(b.text()+" is selected")
                self.d = str(b.text())
        if b.text() == "B":
            if b.isChecked() == True:
                print(b.text()+" is selected")
                self.d = str(b.text())

    def btnstate3(self,b):
        #self.s = "AB"
        if b.text() == "Even":
            if b.isChecked() == True:
                print(b.text()+" is selected")
                self.s = str(b.text())
        if b.text() == "Odd":
            if b.isChecked() == True:
                print(b.text()+" is selected")
                self.s = str(b.text())

    def btnstate4(self,b):
        #self.t = "ABCD"
        if b.text() == "IAT1":
            if b.isChecked() == True:
                print(b.text()+" is selected")
                self.t = str(b.text())
        if b.text() == "IAT2":
            if b.isChecked() == True:
                print(b.text()+" is selected")
                self.t = str(b.text())
        if b.text() == "SEM":
            if b.isChecked() == True:
                print(b.text()+" is selected")
                self.t = str(b.text())

        
    def next_(self):
        print(self.y,"-",self.dep,"-",self.d," ",self.s," sem",self.t," data has been selected for analysis")


class FileUpload(QtGui.QWidget):
    def __init__(self, parent = None):
        super(FileUpload, self).__init__(parent)

        layout = QtGui.QVBoxLayout()
        self.btn = QtGui.QPushButton("Upload File and Analyse")
        self.btn.clicked.connect(self.getfile)
		
        layout.addWidget(self.btn)
        self.le = QtGui.QLabel("            ")
		
        #layout.addWidget(self.le)
        #self.btn1 = QtGui.QPushButton("Analyse")
        layout.setAlignment(QtCore.Qt.AlignCenter)
        #self.btn1.clicked.connect(self.analyse)
        #layout.addWidget(self.btn1)
		
        #self.contents = QTextEdit()
        #layout.addWidget(self.contents)
        self.setLayout(layout)
        self.setWindowTitle("ABCD")
      
    

    def getfile(self):
        fname = QtGui.QFileDialog.getOpenFileName(self, 'Open file', 'C:/Users/nachiket/Desktop/attendance research correlation',"CSV files (*.csv)")
        #self.le.setPixmap(QPixmap(fname))
        print(fname)
        with open(fname,'r') as csvfile:
            readCSV = csv.reader(csvfile,delimiter=',')
            aitat = []
            aitmk = []
            dmbiat = []
            dmbimk = []
            dsat = []
            dsmk = []
            seat = []
            semk = []
            swsat = []
            swsmk = []
            for row in readCSV:
                aitattendance = row[0]
                aitmarks = row[1]
                aitat.append(float(aitattendance))
                aitmk.append(float(aitmarks))
        
                dmbiattendance = row[2]
                dmbimarks = row[3]
                dmbiat.append(float(dmbiattendance))
                dmbimk.append(float(dmbimarks))
        
                dsattendance = row[4]
                dsmarks = row[5]
                dsat.append(float(dsattendance))
                dsmk.append(float(dsmarks))
        
                seattendance = row[6]
                semarks = row[7]
                seat.append(float(seattendance))
                semk.append(float(semarks))
        
                swsattendance = row[8]
                swsmarks = row[9]
                swsat.append(float(swsattendance))
                swsmk.append(float(swsmarks))

        print(aitat)
        print("\n")
        print(aitmk)
        print("\n")
        print(dmbiat)
        print("\n")
        print(dmbimk)
        print("\n")
        print(dsat)
        print("\n")
        print(dsmk)
        print("\n")
        print(seat)
        print("\n")
        print(semk)
        print("\n")
        print(swsat)
        print("\n")
        print(swsmk)
        self.aitattend = aitat
        self.aitmark = aitmk
        self.dmbiattend = dmbiat
        self.dmbimark = dmbimk
        self.dsattend = dsat
        self.dsmark = dsmk
        self.seattend = seat
        self.semark = semk
        self.swsattend = swsat
        self.swsmark = swsmk
        self.ait = []
        self.dmbi = []
        self.ds = []
        self.se = []
        self.sws = []
        self.m1 = st.mean(self.aitattend)
        self.m2 = st.mean(self.aitmark)
        self.m3 = st.mean(self.dmbiattend)
        self.m4 = st.mean(self.dmbimark)
        self.m5 = st.mean(self.dsattend)
        self.m6 = st.mean(self.dsmark)
        self.m7 = st.mean(self.seattend)
        self.m8 = st.mean(self.semark)
        self.m9 = st.mean(self.swsattend)
        self.m10 = st.mean(self.swsmark)
        self.r1 = max(self.aitattend)-min(self.aitattend)
        self.r2 = max(self.aitmark)-min(self.aitmark)
        self.r3 = max(self.dmbiattend)-min(self.dmbiattend)
        self.r4 = max(self.dmbimark)-min(self.dmbimark)
        self.r5 = max(self.dsattend)-min(self.dsattend)
        self.r6 = max(self.dsmark)-min(self.dsmark)
        self.r7 = max(self.seattend)-min(self.seattend)
        self.r8 = max(self.semark)-min(self.semark)
        self.r9 = max(self.swsattend)-min(self.swsattend)
        self.r10 = max(self.swsmark)-min(self.swsmark)
        #joining arrays and normalizing
        for row in range(len(self.aitattend)):
            inait = [(self.aitattend[row]-self.m1)/self.r1,(self.aitmark[row]-self.m2)/self.r2]
            self.ait.append(inait)
            indmb = [(self.dmbiattend[row]-self.m3)/self.r3,(self.dmbimark[row]-self.m4)/self.r4]
            self.dmbi.append(indmb)
            inds = [(self.dsattend[row]-self.m5)/self.r5,(self.dsmark[row]-self.m6)/self.r6]
            self.ds.append(inds)
            inse = [(self.seattend[row]-self.m7)/self.r7,(self.semark[row]-self.m8)/self.r8]
            self.se.append(inse)
            insws = [(self.swsattend[row]-self.m9)/self.r9,(self.swsmark[row]-self.m10)/self.r10]
            self.sws.append(insws)
        self.A = np.array(self.ait)
        self.B = np.array(self.dmbi)
        self.C = np.array(self.ds)
        self.D = np.array(self.se)
        self.E = np.array(self.sws)

        print(len(self.A))
        print(len(self.B))
        print(len(self.C))
        print(len(self.D))
        print(len(self.E))
        
        
        k1 = KMeans(n_clusters=3,random_state=42)
        k1.fit(self.A)
        c1 = k1.cluster_centers_
        l1 = k1.labels_
        
        k2 = KMeans(n_clusters=3,random_state=42)
        k2.fit(self.B)
        c2 = k2.cluster_centers_
        l2 = k2.labels_

        k3 = KMeans(n_clusters=3,random_state=42)
        k3.fit(self.C)
        c3 = k3.cluster_centers_
        l3 = k3.labels_

        k4 = KMeans(n_clusters=3,random_state=42)
        k4.fit(self.D)
        c4 = k4.cluster_centers_
        l4 = k4.labels_

        k5 = KMeans(n_clusters=3,random_state=42)
        k5.fit(self.E)
        c5 = k5.cluster_centers_
        l5 = k5.labels_
        
        fig = plt.figure()
        colors = 10*['r.','g.','b.','c.','k.','y.','m.']
        plot1 = fig.add_subplot(2,3,1)
        for i in range(len(self.A)):
            print("coordinate:",self.A[i],"label:",l1[i])
            plot1.plot(self.A[i][0], self.A[i][1], colors[l1[i]], markersize = 10)
    
        plot1.scatter(c1[:, 0], c1[:, 1], marker = 'x', s = 150, linewidths = 5, zorder = 10)
        plot1.set_xlabel("Attendance")
        plot1.set_ylabel("Marks")
        plot1.set_title("AIT")

        plot2 = fig.add_subplot(2,3,2)
        for i in range(len(self.B)):
            print("coordinate:",self.B[i],"label:",l2[i])
            plot2.plot(self.B[i][0], self.B[i][1], colors[l2[i]], markersize = 10)
    
        plot2.scatter(c2[:, 0], c2[:, 1], marker = 'x', s = 150, linewidths = 5, zorder = 10)
        plot2.set_xlabel("Attendance")
        plot2.set_ylabel("Marks")
        plot2.set_title("DMBI")

        plot3 = fig.add_subplot(2,3,3)
        for i in range(len(self.C)):
            print("coordinate:",self.C[i],"label:",l3[i])
            plot3.plot(self.C[i][0], self.C[i][1], colors[l3[i]], markersize = 10)
    
        plot3.scatter(c3[:, 0], c3[:, 1], marker = 'x', s = 150, linewidths = 5, zorder = 10)
        plot3.set_xlabel("Attendance")
        plot3.set_ylabel("Marks")
        plot3.set_title("DS")

        plot4 = fig.add_subplot(2,3,4)
        for i in range(len(self.D)):
            print("coordinate:",self.D[i],"label:",l4[i])
            plot4.plot(self.D[i][0], self.D[i][1], colors[l4[i]], markersize = 10)
    
        plot4.scatter(c4[:, 0], c4[:, 1], marker = 'x', s = 150, linewidths = 5, zorder = 10)
        plot4.set_xlabel("Attendance")
        plot4.set_ylabel("Marks")
        plot4.set_title("SE")

        plot5 = fig.add_subplot(2,3,5)
        for i in range(len(self.E)):
            print("coordinate:",self.E[i],"label:",l5[i])
            plot5.plot(self.E[i][0], self.E[i][1], colors[l5[i]], markersize = 10)
    
        plot5.scatter(c5[:, 0], c5[:, 1], marker = 'x', s = 150, linewidths = 5, zorder = 10)
        plot5.set_xlabel("Attendance")
        plot5.set_ylabel("Marks")
        plot5.set_title("SWS")
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.show()        

        

if __name__ == '__main__':
    app = QtGui.QApplication([])
    window = MainWindow()
    window.show()
    app.exec()

    
        
        

        
        
