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

def ReedSolomonCodes(block):
    corBytesCount = 28
    polinom  = [168 ,223 ,200,104 ,224, 234 ,108, 180, 110,190, 195, 147, 205,  27, 232,
     201,  21 , 43, 245, 87,  42 ,195 ,212 ,119, 242,  37  , 9 ,123]
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
            # print(corrArr)
    return corrArr[:corBytesCount]

b = [64,196 ,132 , 84 ,196, 196 ,242 ,194  , 4 ,132 , 20  ,37, 34,  16, 236  ,17]
print(ReedSolomonCodes(b))