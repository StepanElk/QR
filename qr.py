import matplotlib.pyplot as plt
import numpy as np

#Уровень М - 15%
#Побайтовое кодирование  - 0100
versions = [128,224,352,512,688,864,992,1232,1456,1728,
            2032,2320,2672,2920,3320,3624,4056,4504,5016,5352,
            5712,6256,6880,7312,8000,8496,9024,9544,10136,10984,
            11640,12328,13048,13800,14496,15312,15936,16816,17728,18672]
version = 0

correctionBytes = [
    10,	16,	26,	18,	24,	16,	18,	22,	22,	26,
    30,	22,	22,	24,	24,	28,	28,	26,	26,	26,
    26,	28,	28,	28,	28,	28,	28,	28,	28,	28,
    28,	28,	28,	28,	28,	28,	28,	28,	28,	28,
]

generatingPolinoms = {
    10 : [	251, 67, 46, 61, 118, 70, 64, 94, 32, 45],
    16 : [	120, 104, 107, 109, 102, 161, 76, 3, 91, 191, 147, 169, 182, 194, 225, 120],
    18 : [215, 234, 158, 94, 184, 97, 118, 170, 79, 187, 152, 148, 252, 179, 5, 98, 96, 153],
    24 : [	229, 121, 135, 48, 211, 117, 251, 126, 159, 180, 169, 152, 192, 226, 228, 218, 111, 0, 117, 232, 87, 96, 227, 21],
    22 : [	210, 171, 247, 242, 93, 230, 14, 109, 221, 53, 200, 74, 8, 172, 98, 80, 219, 134, 160, 105, 165, 231],
    26 : [173, 125, 158, 2, 103, 182, 118, 17, 145, 201, 111, 28, 165, 53, 161, 21, 245, 142, 13, 102, 48, 227, 153, 145, 218, 70],
    28 : [168, 223, 200, 104, 224, 234, 108, 180, 110, 190, 195, 147, 205, 27, 232, 201, 21, 43, 245, 87, 42, 195, 212, 119, 242, 37, 9, 123],
    30 : [	41, 173, 145, 152, 216, 31, 179, 182, 50, 48, 110, 86, 239, 96, 222, 125, 42, 173, 226, 193, 224, 130, 156, 37, 251, 216, 238, 40, 192, 180],
}


blockCount = [
    1,1	,1,	2,2	,4,4,4,5,5,
    5,8	,9,	9,10,10	,11	,13,14,	16,
    17,17,18,20,21,	23,	25,	26,	28,	29,
    31,33,35,37,38,	40,	43,	45,	47,	49]

galoisField = [
    1,	2,	4,	8,	16,	32,	64	,128,29,58,	116,232,205,135,19,	38,
    76,	152,45,	90,	180,117,234,201,143,3,	6,	12,	24,	48,	96,	192,
    157,39,	78,	156,37,	74,	148,53,	106,212,181,119,238,193,159,35,
    70,	140,5,	10,	20,	40,	80,	160,93,	186,105,210,185,111,222,161,
    95,	190,97,	194,153,47,	94,	188,101,202,137,15,	30,	60,	120,240,
    253,231,211,187,107,214,177,127,254,225,223,163,91,	182,113,226,
    217,175,67,	134,17,	34,	68,	136,13,	26,	52,	104,208,189,103,206,
    129,31,	62,	124,248,237,199,147,59,	118,236,197,151,51,	102,204,
    133,23,	46,	92,	184,109,218,169,79,	158,33,	66,	132,21,	42,	84,
    168,77,	154,41,	82,	164,85,	170,73,	146,57,	114,228,213,183,115,
    230,209,191,99,	198,145,63,	126,252,229,215,179,123,246,241,255,
    227,219,171,75,	150,49,	98,	196,149,55,	110,220,165,87,	174,65,
    130,25,	50,	100,200,141,7,	14,	28,	56,	112,224,221,167,83,	166,
    81,	162,89,	178,121,242,249,239,195,155,43,	86,	172,69,	138,9,
    18,	36,	72,	144,61,	122,244,245,247,243,251,235,203,139,11,	22,
    44,	88,	176,125,250,233,207,131,27,	54,	108,216,173,71,	142,1
]

alignmentPatterns={
    2 : [18],7 : [6, 22, 38],12 : [6, 32, 58],17 : [6, 30, 54, 78],22 : [6, 26, 50, 74, 98],27 : [6, 34, 62, 90, 118],32 : [6, 34, 60, 86, 112, 138],37 : [6, 28, 54, 80, 106, 132, 158],
    3 : [22],8 : [6, 24, 42],13 : [6, 34, 62],18 : [6, 30, 56, 82],23 : [6, 30, 54, 78, 102],28 : [6, 26, 50, 74, 98, 122],33: [6, 30, 58, 86, 114, 142],38 : [6, 32, 58, 84, 110, 136, 162],
    4 : [26],9 : [6, 26, 46],14 : [6, 26, 46, 66],19 : [6, 30, 58, 86],24: [6, 28, 54, 80, 106],29 : [6, 30, 54, 78, 102, 126],34 : [6, 34, 62, 90, 118, 146],39 : [6, 26, 54, 82, 110, 138, 166],
    5 : [30],10 : [6, 28, 50],15 : [6, 26, 48, 70],20 : [6, 34, 62, 90],25 : [6, 32, 58, 84, 110],30 : [6, 26, 52, 78, 104, 130],35 : [6, 30, 54, 78, 102, 126, 150],40 : [6, 30, 58, 86, 114, 142, 170],
    6 : [34],11 : [6, 30, 54],16 : [6, 26, 50, 74],21 : [6, 28, 50, 72, 94],26 : [6, 30, 58, 86, 114],31 : [6, 30, 56, 82, 108, 134],36 : [6, 24, 50, 76, 102, 128, 154],
}

versionCode = [
    '000010 011110 100110','010001 011100 111000','110111 011000 000100','101001 111110 000000','001111 111010 111100',
    '001101 100100 011010','101011 100000 100110','110101 000110 100010','010011 000010 011110','011100 010001 011100',
    '111010 010101 100000','100100 110011 100100','000010 110111 011000','000000 101001 111110','100110 101101 000010',
    '111000 001011 000110','011110 001111 111010','001101 001101 100100','101011 001001 011000','110101 101111 011100',
    '010011 101011 100000','010001 110101 000110','110111 110001 111010','101001 010111 111110','001111 010011 000010',
    '101000 011000 101101','001110 011100 010001','010000 111010 010101','110110 111110 101001','110100 100000 001111',
    '010010 100100 110011','001100 000010 110111','101010 000110 001011','111001 000100 010101'
]

class Canvas:
    def __init__(self , size ) :
        self.size = size
        self.arr = [0.5]*size**2
    
    def drawPoint(self , x , y , val):
        self.arr[y*self.size+x] = val
    
    def addSearchPattern(self , cY , cX):
        self.drawRect(cX,cY , -2,3,0,2)
        for row in range(cY-1 , cY+2):          
            for col in range(cX-1 , cX+2):
                self.drawPoint(col,row,1)

        self.drawRect(cX,cY , -3,4,1,3)


    def addAligmentPatterns(self , cX , cY ):
        self.drawRect(cX,cY , -1,2,0,1)
        self.drawPoint(cX,cY,1)
        self.drawRect(cX,cY , -2,3,1,2)

    def addSynchPatterns(self , x1,x2,y2,y3):
        color = 1
        for i in range(x1 , x2 , -1):
            self.drawPoint(i,y2 , color)
            color = (color+1) %2
        color = 0
        for i in range(y2+1 , y3+1):
            self.drawPoint(x2,i , color)
            color = (color+1) %2

    def addVerCode(self , x , y):
        code = versionCode[version-1]
        for char in code:
            if char == ' ':
                x+=1
                y = 0 
                continue
            self.drawPoint(x, y , int(char))
            self.drawPoint(y, x , int(char))
            y+=1

    def addMarkers(self , x1,y1,x2,y2):
        masCode = '101111001111100'

        for i in range(0 , 6):
            self.drawPoint(i , 8, int(masCode[i]))

        self.drawPoint(7 , 8, int(masCode[6]))
        self.drawPoint(8 , 8, int(masCode[7]))
        self.drawPoint(8 , 7, int(masCode[8]))

        x = -4
        for i in range(9 , 15):
            self.drawPoint(8 ,x + i, int(masCode[i]))
            x-=2

        for i in range(0 , 8):
            self.drawPoint(x1+4 , y1 - 4 + i , 0)
            self.drawPoint(x1+5 , y1 + 3 - i , int(masCode[i]))
        self.drawPoint(x1+5 , y1 - 4 , 1)
        
        for i in range(0 , 9):
            self.drawPoint(i , y1-4, 0)
            self.drawPoint(x2-4 ,i, 0)

        for i in range(0 , 8):
            self.drawPoint(x2 - 4 +i , y2+4 , 0)

        for i in range(7 , 15):   
            self.drawPoint(x2 - 11 + i , y2+5 ,  int(masCode[i]))

    def drawRect(self , cX , cY  , r1 , r2 , color  , length):
         for col in range(r1 , r2):      
            self.drawPoint(cX+col,cY-length,color)
            self.drawPoint(cX+col,cY+length,color)
            self.drawPoint(cX-length,cY+col,color)
            self.drawPoint(cX+length,cY+col,color)

    def tryPoint(self , x , y):
        return self.arr[y*self.size+x] == 0.5

    def draw(self):
        array = np.array(self.arr)
        dpi = 50

        fig = plt.figure( figsize = (8,8), dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])  
        ax.set_axis_off()
        ax.imshow(array.reshape(self.size , self.size), cmap='binary', aspect='auto',
                interpolation='nearest')
        plt.show()

def addSpecialFields(string , ver = 0):
    global version
    length = len(string)
    binString  = ''.join('{:0>8}'.format(str(bin(ord(c)))[2:]) for c in string)
    while versions[ver] < len(binString):
        ver +=1

    fieldLength = 8 if ver<=9 else 16
    dataLength = bin(length)[2:]
    if len(dataLength) < fieldLength:
        dataLength = '0'*(fieldLength-len(dataLength))+dataLength

    version = ver+1
    return '0100'+dataLength+binString

def formateLength(binString):
    fillBytes = '1110110000010001'
    if(len(binString)%8 != 0):
        binString += (8 - len(binString)%8)*'0'
    diff = (versions[version-1] - len(binString))/len(fillBytes)
    binString += fillBytes * int(diff)
    if (len(binString) < versions[version-1]):
        binString += fillBytes[:8]
    return binString

def findBlockCount():
    block = blockCount[version-1]
    dataLength = versions[version-1]//8
    blockLength = dataLength // block
    ost = dataLength % block
    return blockLength , ost , block

def fillBlock(data , blockLength , ost , block):
    baseArr = []
    i = 0
    while i < len(data):
        baseArr.append(int(data[i : i+8] , 2))
        i+=8
    arr = np.array(baseArr)
    if (ost == 0 ):
        return arr.reshape(block , blockLength) , None
    else:
        mainArr = np.array([int(x) for x in baseArr[:blockLength*(block-ost)]])
        mainArr = mainArr.reshape(block-ost , blockLength)
        ostArr = np.array([int(x) for x in baseArr[blockLength*(block-ost):]])
        ostArr = ostArr.reshape(ost , blockLength+1)
        return mainArr , ostArr


def ReedSolomonCodes(block):
    corBytesCount = correctionBytes[version -1 ]
    polinom  = generatingPolinoms[corBytesCount]
    corrArr = []
    corrArr = list(block[:])
    if(len(block) < corBytesCount):
        zeros = [0]*(corBytesCount-len(block))
        corrArr += zeros
    cicleCount = len(block)
    for _ in range(cicleCount):
        A = corrArr.pop(0)
        corrArr.append(0)
        if A == 0 :
            continue
        B = galoisField.index(A)
        polCopy = polinom[:]
        for j in range(len(polinom)):
            polCopy[j] +=  B
            if(polCopy[j] > 254):
                polCopy[j] = polCopy[j]%255
            polCopy[j] = galoisField[polCopy[j]]
            corrArr[j] = corrArr[j]^polCopy[j]
    return corrArr[:corBytesCount]

def combineBlocks(data , corr):
    dataCount = len(data)
    stream = []
    for index in range(len(data[0])):
        for row in range(dataCount):
            if(data[row][index]!=None):
                stream.append(data[row][index])
    for index in range(len(corr[0])):
        for row in range(dataCount):
            stream.append(corr[row][index])
    return np.array(stream)

def drawField():
    size = 21 if version==1 else alignmentPatterns[version][-1]+7
    canvas = Canvas(size)
    canvas.drawRect(4,4,-4,4,0,3)
    canvas.addSearchPattern(3,3)
    canvas.addSearchPattern(size - 4,3)
    canvas.addSearchPattern(3,size - 4)
    canvas.addMarkers(3,size - 4 , size - 4,3)

    if version > 1:
        aligmentPoints = alignmentPatterns[version]
        coordinates = []
        for i in aligmentPoints:
            for j in aligmentPoints:
                coordinates.append((i , j))
        if(version > 6):
            coordinates.pop(0)
            coordinates.pop(len(aligmentPoints)-2)
            coordinates.pop(-(len(aligmentPoints)))
        for pair in coordinates:
            canvas.addAligmentPatterns(pair[0] , pair[1])

    canvas.addSynchPatterns(size - 9 , 6 ,6,size-8)

    if version >6 : 
        canvas.addVerCode(size - 11 , 0)

    return canvas

def drawData(canvas , stream):
    print(stream)
    stream = ''.join('{:0>8}'.format(str(bin(c))[2:]) for c in stream).replace('b','')
    print(stream)
    size = canvas.size
    i = 0
    orderCol = 0
    orderRow  = -1
    col = size-1
    row = size-1
    while i < len(stream):
        tempCol = col - orderCol
        index = row*size+tempCol
        mask = (size  - tempCol)%3 == 0
        if(tempCol==6):
            col-=1
            continue 
        if canvas.arr[index] == 0.5:
            val = int(stream[i]) if not mask else ((int(stream[i]) +1 )%2)
            canvas.arr[index] = val
            i+=1
        orderCol = (orderCol+1)%2
        if (orderCol == 0):
            row+= orderRow
            
        if (row < 0 ):
            row = 0
            orderRow = 1
            col -=2
        if (row >= size ):
            row = size - 1
            orderRow = -1
            col -=2   
        if(row == 0 and col == 0 ):
            break

def conc(data , corr):
    dataCount = len(data)
    stream = []
    for index in range(dataCount):
        for row in range(len(data[0])):
            if(data[index][row]!=None):
                stream.append(data[index][row])
    for index in range(dataCount ):
        for row in range(len(corr[0])):
            stream.append(corr[index][row])
    return np.array(stream)

def run():
    example = 'habr'
    example = u"Hello word!!!"
    bin = addSpecialFields(example)
    bin = formateLength(bin)
    print(bin)
    blockLength , ost , block = findBlockCount()
    mainArr , ostArr = fillBlock(bin , blockLength , ost , block)
    corrBytes = []

    for block in mainArr:
        corrBytes.append(ReedSolomonCodes(block)) 
    if (ost!=0):
        mainArr = np.c_[mainArr , [None]*len(mainArr)]
        for block in ostArr:
            corrBytes.append(ReedSolomonCodes(block)) 
    data = mainArr if ost==0 else np.concatenate((mainArr , ostArr))
    stream = conc(data , corrBytes)
    canvas = drawField()
    print(version)
    drawData(canvas , stream)
    canvas.draw()
run()