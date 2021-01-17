import numpy as np
from PIL import Image

#1번
#(a) 정사각형 이미지의 각 픽셀의 명도값을 저장한 square matrix A 만들기
def get_image(path,k):
    image = Image.open(path).convert('L')
    image = image.crop((0,0,k,k))
    A = np.array(image, 'int64')
    return A
k=512
image = Image.open("Lena.jpg").convert('L')
image = image.crop((0,0,k,k))
image.save("../Project1_results/Lena_edited.jpg")

#(b) nomalized Haar matrix 정의하기
def HaarMatrix(n):
    if n==1: h=np.array([[1]], 'int64')
    else: 
        temp1= np.array([[1],[1]] ,'int64')
        h_l= np.kron(HaarMatrix(n//2), temp1)
 
        temp2 = np.array([[1],[-1]],'int64')
        h_r = np.kron(np.identity(n//2), temp2)

        h = (np.hstack((h_l, h_r)))
    return h

def normalize_column(matrix) :
    transposed_matrix = matrix.T
    normalize_row(transposed_matrix)
    return transposed_matrix.T

def normalize_row(matrix):
    for i in range(len(matrix)) :
        sum = 0
        for j in range(len(matrix[i])):
            sum += (matrix[i][j]**2);
        sum = (sum ** (1/2))
        matrix[i] = matrix[i] / sum 

def normalized_Haarmatrix(n):
    return normalize_column(HaarMatrix(n))

#(c) 2-D Discrete Haar Wavelet Transform (DHWT)을 B=H^TAH로 구현한다.
def DHWT(A):
    H= normalized_Haarmatrix(len(A))
    B= np.dot(np.dot(H.T, A),H)
    return B

#(d) B의 좌측 상단 코너의 k×k를 제외한 모든 요소의 값을 0으로 바꾼 n×n B hat을 만든다.
def compression(B,k): #B와 k를 입력하면 B hat 반환
    n  = len(B)
    B_hat = np.zeros((n,n))
    for i in range(k):
        for j in range(k):
            B_hat[i][j] = B[i][j]
    return B_hat

#(e) A hat = H Bhat H^T 생성 (복원)
def decompression(B_hat):
    H = normalized_Haarmatrix(len(B_hat))
    return np.dot(np.dot(H, B_hat), H.T)
 
#(f) A hat 이미지 띄우고 저장하기
def makeAtoAhat(path, name, k):
    image = get_image(path,512)
    dhwt_image = DHWT(image)
    print(image) #B 행렬 내용 확인
    print("--------")
    compress_image = compression(dhwt_image, k ) #Bhat
    decompress_image = decompression(compress_image) #Ahat
    print(decompress_image) #Ahat 행렬 내용 확인
    image2 = Image.fromarray(decompress_image.astype('uint8'), 'L') #Ahat 이미지로 변환
    image2.show() #Ahat 이미지 띄우기
    image2.save("../Project1_results/ %s k = %d.jpg" % (name, k))

makeAtoAhat("Lena.jpg", "1(f)", 64)

#2번
#(a) 1번에서 작성한 코드에서 k 값이 증가할 때의 이미지 확인하기
for i in range(1,9,1):
    k=2**i
    makeAtoAhat("Lena.jpg", "2(a)", k)


#(b) (a)의 사항을 high frequency와 low frequency image에 적용해보기

#잘린 모습 확인
k=512
image = Image.open("high_frequency.jpg").convert('L')
image = image.crop((0,0,k,k))
image.save("../Project1_results/high_edited.jpg")
image = Image.open("low_frequency.jpg").convert('L')
image = image.crop((0,0,k,k))
image.save("../Project1_results/low_edited.jpg")

#high freqency
for i in range(1,9,1):
    k=2**i
    makeAtoAhat("high_frequency.jpg", "2(b)_high", k)

for i in range(1,9,1):
    k=2**i
    makeAtoAhat("low_frequency.jpg", "2(b)_low", k)

#3번
#(c) Hl^T Hl A Hl^T Hl + Hl^T Hl A Hh^T Hh + Hh^T Hh A Hl^T Hl + Hh^T Hh A Hh^T Hh 의 각 항들을 이미지로 띄우기
# make Hl Hh
def devide(matrix):
    n = len(matrix[0])
    half = len(matrix) // 2
    hl = np.zeros((half, n))
    hh = np.zeros((half, n))
    for i in range(half):
        for j in range(n):
            hl[i][j] = matrix[i][j]
            hh[i][j] = matrix[i + half][j]
    return hl, hh

image=get_image("Lena.jpg",512)
haar = normalized_Haarmatrix(len(image))
hl, hh = devide(haar.T)
print(hl.shape)
temp = np.dot(hl.T, hl)
temp2 = np.dot(hh.T, hh)
a = np.dot(np.dot(temp,image),temp)
b = np.dot(np.dot(temp,image),temp2)
c = np.dot(np.dot(temp2,image),temp)
d = np.dot(np.dot(temp2,image), temp2)

image3 = Image.fromarray(a.astype('uint8'), 'L')
image3.show()
image3.save("../Project1_results/3(c)-1.jpg")
image3 = Image.fromarray(b.astype('uint8'), 'L')
image3.show()
image3.save("../Project1_results/3(c)-2.jpg")
image3 = Image.fromarray(c.astype('uint8'), 'L')
image3.show()
image3.save("../Project1_results/3(c)-3.jpg")
image3 = Image.fromarray(d.astype('uint8'), 'L')
image3.show()
image3.save("../Project1_results/3(c)-4.jpg")

sum_arr = a+b+c+d #각 항들의 총합이 다시 원본 이미지, 즉 A행렬이 되는지 확인하기
image3 = Image.fromarray(sum_arr.astype('uint8'),'L')
image3.show()
image3.save("../Project1_results/3(c)-sum.jpg")

#(d) Hl^T Hl A Hl^T Hl = Hll^T Hll A Hll^T Hll + Hll^T Hll A Hlh^T Hlh + Hlh^T Hlh A Hll^T Hl + Hlh^T Hlh A Hlh^T Hlh
#의 각 항들을 이미지로 띄우기

hll, hlh = devide(hl)

print(hl.shape)
temp = np.dot(hll.T, hll)
temp2 = np.dot(hlh.T, hlh)

a_ = np.dot(np.dot(temp,image),temp)
b_ = np.dot(np.dot(temp,image),temp2)
c_ = np.dot(np.dot(temp2,image),temp)
d_ = np.dot(np.dot(temp2,image), temp2)

image4 = Image.fromarray(a_.astype('uint8'), 'L')
image4.show()
image4.save("../Project1_results/3(d)-1.jpg")
image4 = Image.fromarray(b_.astype('uint8'), 'L')
image4.show()
image4.save("../Project1_results/3(d)-2.jpg")
image4 = Image.fromarray(c_.astype('uint8'), 'L')
image4.show()
image4.save("../Project1_results/3(d)-3.jpg")
image4 = Image.fromarray(d_.astype('uint8'), 'L')
image4.show()
image4.save("../Project1_results/3(d)-4.jpg")

sum_arr = a_+b_+c_+d_ #각 항들의 총합이 다시 원본 이미지, 즉 Hl^T Hl A Hl^T Hl가 되는지 확인하기
image4 = Image.fromarray(sum_arr.astype('uint8'),'L')
image4.show()
image4.save("../Project1_results/3(d)-sum.jpg")
