# -*- coding: utf-8 -*-

import sys
from random import randint
from PyQt4 import QtGui,QtCore
from functools import partial  
from rsa import newkeys
from timeit import Timer

__ip = [  
    58,50,42,34,26,18,10,2,60,52,44,36,28,20,12,4,  
    62,54,46,38,30,22,14,6,64,56,48,40,32,24,16,8,  
    57,49,41,33,25,17, 9,1,59,51,43,35,27,19,11,3,  
    61,53,45,37,29,21,13,5,63,55,47,39,31,23,15,7,  
]  
__ip1 = [  
    40,8,48,16,56,24,64,32,39,7,47,15,55,23,63,31,  
    38,6,46,14,54,22,62,30,37,5,45,13,53,21,61,29,  
    36,4,44,12,52,20,60,28,35,3,43,11,51,19,59,27,  
    34,2,42,10,50,18,58,26,33,1,41, 9,49,17,57,25,  
]  
__e = [  
    32, 1, 2, 3, 4, 5,  
    4 , 5, 6, 7, 8, 9,  
    8 , 9,10,11,12,13,  
    12,13,14,15,16,17,  
    16,17,18,19,20,21,  
    20,21,22,23,24,25,  
    24,25,26,27,28,29,  
    28,29,30,31,32, 1,  
]  
__p = [  
    16, 7,20,21,29,12,28,17,  
    1 ,15,23,26, 5,18,31,10,  
    2 ,8 ,24,14,32,27, 3, 9,  
    19,13,30, 6,22,11, 4,25,  
]  
__rp = [
    9, 17, 23, 31,
    13, 28, 2, 18,
    24, 16, 30, 6,
    26, 20, 10, 1,
    8, 14, 25, 3,
    4, 29, 11, 19,
    32, 12, 22, 7,
    5, 27, 15, 21 
]
__s = [  
[  
    0xe,0x4,0xd,0x1,0x2,0xf,0xb,0x8,0x3,0xa,0x6,0xc,0x5,0x9,0x0,0x7,  
    0x0,0xf,0x7,0x4,0xe,0x2,0xd,0x1,0xa,0x6,0xc,0xb,0x9,0x5,0x3,0x8,  
    0x4,0x1,0xe,0x8,0xd,0x6,0x2,0xb,0xf,0xc,0x9,0x7,0x3,0xa,0x5,0x0,  
    0xf,0xc,0x8,0x2,0x4,0x9,0x1,0x7,0x5,0xb,0x3,0xe,0xa,0x0,0x6,0xd,  
],  
[  
    0xf,0x1,0x8,0xe,0x6,0xb,0x3,0x4,0x9,0x7,0x2,0xd,0xc,0x0,0x5,0xa,  
    0x3,0xd,0x4,0x7,0xf,0x2,0x8,0xe,0xc,0x0,0x1,0xa,0x6,0x9,0xb,0x5,  
    0x0,0xe,0x7,0xb,0xa,0x4,0xd,0x1,0x5,0x8,0xc,0x6,0x9,0x3,0x2,0xf,  
    0xd,0x8,0xa,0x1,0x3,0xf,0x4,0x2,0xb,0x6,0x7,0xc,0x0,0x5,0xe,0x9,  
],  
[  
    0xa,0x0,0x9,0xe,0x6,0x3,0xf,0x5,0x1,0xd,0xc,0x7,0xb,0x4,0x2,0x8,  
    0xd,0x7,0x0,0x9,0x3,0x4,0x6,0xa,0x2,0x8,0x5,0xe,0xc,0xb,0xf,0x1,  
    0xd,0x6,0x4,0x9,0x8,0xf,0x3,0x0,0xb,0x1,0x2,0xc,0x5,0xa,0xe,0x7,  
    0x1,0xa,0xd,0x0,0x6,0x9,0x8,0x7,0x4,0xf,0xe,0x3,0xb,0x5,0x2,0xc,  
],  
[  
    0x7,0xd,0xe,0x3,0x0,0x6,0x9,0xa,0x1,0x2,0x8,0x5,0xb,0xc,0x4,0xf,  
    0xd,0x8,0xb,0x5,0x6,0xf,0x0,0x3,0x4,0x7,0x2,0xc,0x1,0xa,0xe,0x9,  
    0xa,0x6,0x9,0x0,0xc,0xb,0x7,0xd,0xf,0x1,0x3,0xe,0x5,0x2,0x8,0x4,  
    0x3,0xf,0x0,0x6,0xa,0x1,0xd,0x8,0x9,0x4,0x5,0xb,0xc,0x7,0x2,0xe,  
],  
[  
    0x2,0xc,0x4,0x1,0x7,0xa,0xb,0x6,0x8,0x5,0x3,0xf,0xd,0x0,0xe,0x9,  
    0xe,0xb,0x2,0xc,0x4,0x7,0xd,0x1,0x5,0x0,0xf,0xa,0x3,0x9,0x8,0x6,  
    0x4,0x2,0x1,0xb,0xa,0xd,0x7,0x8,0xf,0x9,0xc,0x5,0x6,0x3,0x0,0xe,  
    0xb,0x8,0xc,0x7,0x1,0xe,0x2,0xd,0x6,0xf,0x0,0x9,0xa,0x4,0x5,0x3,  
],  
[  
    0xc,0x1,0xa,0xf,0x9,0x2,0x6,0x8,0x0,0xd,0x3,0x4,0xe,0x7,0x5,0xb,  
    0xa,0xf,0x4,0x2,0x7,0xc,0x9,0x5,0x6,0x1,0xd,0xe,0x0,0xb,0x3,0x8,  
    0x9,0xe,0xf,0x5,0x2,0x8,0xc,0x3,0x7,0x0,0x4,0xa,0x1,0xd,0xb,0x6,  
    0x4,0x3,0x2,0xc,0x9,0x5,0xf,0xa,0xb,0xe,0x1,0x7,0x6,0x0,0x8,0xd,  
],  
[  
    0x4,0xb,0x2,0xe,0xf,0x0,0x8,0xd,0x3,0xc,0x9,0x7,0x5,0xa,0x6,0x1,  
    0xd,0x0,0xb,0x7,0x4,0x9,0x1,0xa,0xe,0x3,0x5,0xc,0x2,0xf,0x8,0x6,  
    0x1,0x4,0xb,0xd,0xc,0x3,0x7,0xe,0xa,0xf,0x6,0x8,0x0,0x5,0x9,0x2,  
    0x6,0xb,0xd,0x8,0x1,0x4,0xa,0x7,0x9,0x5,0x0,0xf,0xe,0x2,0x3,0xc,  
],  
[  
    0xd,0x2,0x8,0x4,0x6,0xf,0xb,0x1,0xa,0x9,0x3,0xe,0x5,0x0,0xc,0x7,  
    0x1,0xf,0xd,0x8,0xa,0x3,0x7,0x4,0xc,0x5,0x6,0xb,0x0,0xe,0x9,0x2,  
    0x7,0xb,0x4,0x1,0x9,0xc,0xe,0x2,0x0,0x6,0xa,0xd,0xf,0x3,0x5,0x8,  
    0x2,0x1,0xe,0x7,0x4,0xa,0x8,0xd,0xf,0xc,0x9,0x0,0x3,0x5,0x6,0xb,  
],  
]  
__k1 = [  
    57,49,41,33,25,17, 9,  
    1 ,58,50,42,34,26,18,  
    10, 2,59,51,43,35,27,  
    19,11, 3,60,52,44,36,  
    63,55,47,39,31,23,15,  
    7 ,62,54,46,38,30,22,  
    14, 6,61,53,45,37,29,  
    21,13, 5,28,20,12, 4,  
]  

__rk1 = [
   8, 16, 24, 56, 52, 44, 36,  0,
   7, 15, 23, 55, 51, 43, 35,  0,
   6, 14, 22, 54, 50, 42, 34,  0,
   5, 13, 21, 53, 49, 41, 33,  0,
   4, 12, 20, 28, 48, 40, 32,  0,
   3, 11, 19, 27, 47, 39, 31,  0,
   2, 10, 18, 26, 46, 38, 30,  0,
   1, 9,  17, 25, 45, 37, 29,  0 
]
__k2 = [  
    14,17,11,24, 1, 5, 3,28,  
    15, 6,21,10,23,19,12, 4,  
    26, 8,16, 7,27,20,13, 2,  
    41,52,31,37,47,55,30,40,  
    51,45,33,48,44,49,39,56,  
    34,53,46,42,50,36,29,32,  
]  
__rk2 = [
   5, 24,  7, 16,  6, 10, 20, 18,
   0, 12,  3, 15, 23,  1,  9, 19,
   2,  0, 14, 22, 11,  0, 13,  4,
   0, 17, 21,  8, 47, 31, 27, 48,
  35, 41,  0, 46, 28,  0, 39, 32,
  25, 44,  0, 37, 34, 43, 29, 36,
  38, 45, 33, 26, 42,  0, 30, 40
]
__k0 = [  
  1,2,4,6,8,10,12,14,15,17,19,21,23,25,27,28
]  
__hex_bin = {  
    '0':'0000','1':'0001','2':'0010','3':'0011',  
    '4':'0100','5':'0101','6':'0110','7':'0111',  
    '8':'1000','9':'1001','a':'1010','b':'1011',  
    'c':'1100','d':'1101','e':'1110','f':'1111',  
    ' ':'0000'  
}  

__re = lambda t, s: ''.join(s[i-1] for i in t)  

__IP = partial(__re, __ip)  
__IP1 = partial(__re, __ip1)  
__E = partial(__re, __e)  
__P = partial(__re, __p)  
__RP = partial(__re, __rp)
__K1 = partial(__re, __k1)  
__RK1 = partial(__re, __rk1)
__K2 = partial(__re, __k2)  
__RK2 = partial(__re, __rk2)

__B = lambda s: ''.join(__hex_bin[w]  for w in ''.join('%2x' % ord(w) for w in s)) 
__DB = lambda s: ''.join(chr(int(s[i:i+8], 2)) for i in range(0, len(s), 8))  
__HB = lambda s: ''.join(__hex_bin[w] for w in s)
__S =  lambda s: ''.join(__hex_bin['%x' % __s[i][int(s[i*6]+s[i*6+5], 2)*16 + int(s[i*6+1:i*6+5], 2)]] for i in range(8))  
__SUB = lambda s, i: __hex_bin['%x' % __s[i][int(s[0]+s[5], 2)*16 + int(s[1:5], 2)]]  
__F = lambda s, k: ''.join('0' if s[i]==k[i] else '1' for i in range(len(s)))  
__K0 =  lambda  k: map(__K2, (k[__k0[i]:28]+k[0:__k0[i]] + k[__k0[i]+28:56]+k[28:__k0[i]+28] for i in range(16)))  
__K = lambda  k: __K0(__K1(k))  

def des_key(key):  
    key = ''.join(__hex_bin[w] for w in key)  
    return __K(key)  

def __code(s, k, rnd=16):  
    if rnd == 16: s = __IP(s)  
    l, r = s[0:32], s[32:64]  
    for i in range(rnd):  
        r_t = r  
        r = __E(r)  
        r = __F(r, k[i])  
        r = __S(r)  
        r = __P(r)  
        r = __F(r, l)  
        l = r_t  
    return __IP1(r+l) if rnd==16 else r+l 
    
def des_en(s, k, rnd=16):  
    a = ''  
    if rnd == 16:
        s += ' ' * ((8-len(s)%8)%8)  
        for i in range(0, len(s), 8):  
            before = __B(s[i:i+8])  
            after = __code(before, k)  
            a += '%16x' % int(after, 2)  
    else :
        before = __HB(s) 
	after = __code(before, k, rnd)
        a += '%16x' % int(after,2)
    return ''.join(w if w!=' ' else '0' for w in a)  

def des_de(s, k, rnd=16):  
    a = ''  
    #s.lower()  
    for i in range(0, len(s), 16):  
        before = ''.join(__hex_bin[s[j]] for j in range(i, i+16))  
	after = __code(before, k[0:rnd][::-1], rnd)  
        a += __DB(after)  
    return a.rstrip()  

def crack6(pt, et, check):
    pl = [[(__HB(x),__HB(y)) for (x,y) in z] for z in pt]
    en = [[(__HB(x), __HB(y)) for (x,y) in z] for z in et]

    en = [[(x[32:64]+x[0:32],y[32:64]+y[0:32]) for (x,y) in z] for z in en]
    ts = [[__E(x[0:32]) for (x,y) in z] for z in en]
    si = [[__F(__E(x[0:32]),__E(y[0:32])) for (x,y) in z] for z in en]
    r6 = [[__F(x[32:64], y[32:64]) for (x,y) in z] for z in en]
    phi = [__HB('0400000000'), __HB('0600000000')]
    so = [[__RP(__F(x,p)) for x in z] for (z,p) in zip(r6, phi)]

    jm = [[0 for x in range(64)] for x in range(8)]
    ki = [1, 0, 1, 1, 0, 0, 0, 0]
    lens = len(pt[0])
    for i in range(lens):
        for j in range(64):
	    x = __HB('%2x' % j)[2:8]
	    for k in range(8):
                p = ki[k]
		y = __F(x,si[p][i][k*6:k*6+6])
		z = __F(__SUB(x,k), __SUB(y,k))
		if z == so[p][i][k*4:k*4+4]:
			t = int(ts[p][i][k*6:k*6+6],2)
			jm[k][j^t] += 1

    k,s = '', [43,36,33,9,61,37,28,7]
    for i in range(8):
	    m = 0
	    for j in range(64):
		if jm[i][j] > jm[i][m]:
	            m = j
            k += __HB('%2x' % m)[2:8]

    k,l = __RK2(k), __k0[5] 
    k = __RK1(k[28-l:28]+k[0:28-l]+k[56-l:56]+k[28:56-l])
    for i in range(256):
            x,y = __HB('%2x' % i), list(k)
            for j in range(8):
        	    y[s[j]-1] = x[j]
            y = '%16x' % int(''.join(y), 2)
	    y = ''.join(w if w != ' ' else '0' for w in y)
            key = des_key(y)
            if check == des_en("shit", key) : 
		    return y

def test_gen6(num = 480):
    x = [__HB('%16x' % randint(0, (1<<64)-1)) for  t in range(num)]
    d = [__HB('4008000004000000'), __HB('0000401006000000')]
    y = [['%16x' % int(__F(a,b), 2) for a in x] for b in d]
    return [[('%16x' % int(a,2), b) for (a,b) in zip(x,c)] for c in y] 

def key_test(k):
    k = k[:16]
    if k == '0'*16 : return 0
    if k == 'f'*16 : return 0
    if k == '0'*8+'f'*8 : return 0
    if k == 'f'*8+'0'*8 : return 0
    return 1

def rsa_en(s, k, f=pow):
    before = reduce(lambda x,y:x*256+ord(y), s, 0)
    return f(before, k.e, k.n)
   
def rsa_de(s, k, f=pow):
    before = '%x' % f(s, k.d, k.n)
    return ''.join(chr(int(before[i:i+2],16)) for i in range(0, len(before), 2)) 


#pub, pri = newkeys(512)
#s = raw_input("str:\n")
#e = rsa_en(s, pub)
#print '%x\n%r' % (e , len('%x' % e))
#print rsa_de(e, pri)

#crk = test_gen3(5)
###k = des_key("0000000000000000")
#k = des_key("1a624c89520fec46")
##print "1a624c89520fec46"
#print crk 
#en = [(des_en(x,k,3), des_en(y,k,3)) for (x,y) in crk] 
#print en
#print crack3(crk, en)

#k = des_key("1a624c89520fec46")
#print "1a624c89520fec46"
#crk = test_gen6(240)
#en = [[(des_en(x,k,6), des_en(y,k,6)) for (x,y) in z] for z in crk]
#print crack6(crk, en, des_en("shit",k))

class InputDialog(QtGui.QWidget):
    def __init__(self, parent = None):
        QtGui.QWidget.__init__(self)
        self.setGeometry(300, 300, 600, 500)
        self.setWindowTitle(u'密码学课程设计')

	tabs = QtGui.QTabWidget(self)
	tab_des = QtGui.QWidget()
	tab_crk = QtGui.QWidget()
	tab_rsa = QtGui.QWidget()

        #DES UI
	tabs.addTab(tab_des, u'DES加解密')

	hbox = QtGui.QHBoxLayout()
	self.des_input = QtGui.QTextEdit()
	self.des_output = QtGui.QTextEdit()
	self.des_key = QtGui.QLineEdit()
	vbox = QtGui.QVBoxLayout()
	vbox.addWidget(QtGui.QLabel(u'输入:'))
	vbox.addWidget(self.des_input)
	h2box = QtGui.QHBoxLayout()
	h2box.addWidget(QtGui.QLabel(u'密钥:'))
	h2box.addWidget(self.des_key)
        vbox.addLayout(h2box)	
	hbox.addLayout(vbox)

	vbox = QtGui.QVBoxLayout()
	vbox.addWidget(QtGui.QLabel(u'输出:'))
	vbox.addWidget(self.des_output)
	h2box = QtGui.QHBoxLayout()
	en_button = QtGui.QPushButton(u'加密')
	de_button = QtGui.QPushButton(u'解密')
	h2box.addStretch(1)
	h2box.addWidget(en_button)
	h2box.addWidget(de_button)
	vbox.addLayout(h2box)

	hbox.addLayout(vbox)
	tab_des.setLayout(hbox)

        self.connect(en_button, QtCore.SIGNAL('clicked()'), self.endes)  
        self.connect(de_button, QtCore.SIGNAL('clicked()'), self.dedes)  

	tabs.addTab(tab_crk, u'DES差分攻击')

	tabs.addTab(tab_rsa, u'RSA加解密')

	vbox = QtGui.QVBoxLayout()
        vbox.addWidget(tabs)
        
        self.setLayout(vbox)
    
    def endes(self) :
        key = self.des_key.text().toLocal8Bit()
	if key_test(key) == 0: 
            print "shit"
	x = self.des_input.toPlainText().toLocal8Bit()
	key = des_key(key)
	y = des_en(x, key)
	self.des_output.setText(unicode(y))

    def dedes(self) :
        key = self.des_key.text().toLocal8Bit()
	if key_test(key) == 0: 
            print "shit"
	x = self.des_input.toPlainText().toLocal8Bit()
	key = des_key(key)
	y = des_de(x, key)
	self.des_output.setText(unicode(y, 'utf-8'))



if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    icon = InputDialog()
    icon.show()
    sys.exit(app.exec_())
