import base64
import bz2
import socket

import cv2
import numpy as np
import pytesseract
import matplotlib.pyplot as plt
from PIL import Image

letter=np.zeros(256)


def CalcImageHash(FileName):
    image = cv2.imread(FileName)  # Прочитаем картинку
    resized = cv2.resize(image, (8, 8), interpolation=cv2.INTER_AREA)  # Уменьшим картинку
    gray_image = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)  # Переведем в черно-белый формат
    avg = gray_image.mean()  # Среднее значение пикселя
    ret, threshold_image = cv2.threshold(gray_image, avg, 255, 0)  # Бинаризация по порогу

    # Рассчитаем хэш
    _hash = ""
    for x in range(8):
        for y in range(8):
            val = threshold_image[x, y]
            if val == 255:
                _hash = _hash + "1"
            else:
                _hash = _hash + "0"

    return _hash


def CompareHash(hash1, hash2):
    l = len(hash1)
    i = 0
    count = 0
    while i < l:
        if hash1[i] != hash2[i]:
            count = count + 1
        i = i + 1
    return count

def letters_extract(image_file: str, out_size=28):
    img = cv2.imread(image_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    img_erode = cv2.erode(thresh, np.ones((3, 3), np.uint8), iterations=1)

    # Get contours
    contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    output = img.copy()

    letters = []
    for idx, contour in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(contour)
        # print("R", idx, x, y, w, h, cv2.contourArea(contour), hierarchy[0][idx])
        # hierarchy[i][0]: the index of the next contour of the same level
        # hierarchy[i][1]: the index of the previous contour of the same level
        # hierarchy[i][2]: the index of the first child
        # hierarchy[i][3]: the index of the parent
        if hierarchy[0][idx][3] == 0:
            cv2.rectangle(output, (x, y), (x + w, y + h), (70, 0, 0), 1)
            letter_crop = gray[y:y + h, x:x + w]
            # print(letter_crop.shape)

            # Resize letter canvas to square
            size_max = max(w, h)
            letter_square = 255 * np.ones(shape=[size_max, size_max], dtype=np.uint8)
            if w > h:
                # Enlarge image top-bottom
                # ------
                # ======
                # ------
                y_pos = size_max//2 - h//2
                letter_square[y_pos:y_pos + h, 0:w] = letter_crop
            elif w < h:
                # Enlarge image left-right
                # --||--
                x_pos = size_max//2 - w//2
                letter_square[0:h, x_pos:x_pos + w] = letter_crop
            else:
                letter_square = letter_crop

            # Resize letter to 28x28 and add letter and its X-coordinate
            letters.append((x, w, cv2.resize(letter_square, (out_size, out_size), interpolation=cv2.INTER_AREA)))

    # Sort array in place by X-coordinate
    letters.sort(key=lambda x: x[0], reverse=False)
    lett="01234567890ABCDEF"
    ans=""
    for i in letters:
        hashes=np.zeros(len(lett))
        cv2.imwrite("buf.png",i[2])
        imhash=CalcImageHash("buf.png")
        for j in range(len(lett)):
            hashes[j]+=CompareHash(imhash,CalcImageHash("data/"+lett[j]+'.png'))
        ans+=lett[hashes.argmin()]
    return ans


def letters_extract2(image_file: str, sss,out_size=28):
    img = cv2.imread(image_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    img_erode = cv2.erode(thresh, np.ones((3, 3), np.uint8), iterations=1)

    # Get contours
    contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    output = img.copy()

    letters = []
    for idx, contour in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(contour)
        # print("R", idx, x, y, w, h, cv2.contourArea(contour), hierarchy[0][idx])
        # hierarchy[i][0]: the index of the next contour of the same level
        # hierarchy[i][1]: the index of the previous contour of the same level
        # hierarchy[i][2]: the index of the first child
        # hierarchy[i][3]: the index of the parent
        if hierarchy[0][idx][3] == 0:
            cv2.rectangle(output, (x, y), (x + w, y + h), (70, 0, 0), 1)
            letter_crop = gray[y:y + h, x:x + w]
            # print(letter_crop.shape)

            # Resize letter canvas to square
            size_max = max(w, h)
            letter_square = 255 * np.ones(shape=[size_max, size_max], dtype=np.uint8)
            if w > h:
                # Enlarge image top-bottom
                # ------
                # ======
                # ------
                y_pos = size_max//2 - h//2
                letter_square[y_pos:y_pos + h, 0:w] = letter_crop
            elif w < h:
                # Enlarge image left-right
                # --||--
                x_pos = size_max//2 - w//2
                letter_square[0:h, x_pos:x_pos + w] = letter_crop
            else:
                letter_square = letter_crop

            # Resize letter to 28x28 and add letter and its X-coordinate
            letters.append((x, w, cv2.resize(letter_square, (out_size, out_size), interpolation=cv2.INTER_AREA)))

    # Sort array in place by X-coordinate
    letters.sort(key=lambda x: x[0], reverse=False)
    for i in range(len(letters)):
        cv2.imwrite("data/"+sss[i]+".png",letters[i][2])




def recon(xx):
    x = []
    y = []
    s = bz2.decompress(base64.b85decode(xx)).decode("utf-8").strip().split('\n')

    s = s[1:]
    s = list(map(lambda x: x[8:-1], s))
    s = list(map(lambda x: [x.split(';')[0].replace('{', '').replace('}', '').replace('^', '**'),
                            x.split(';')[1].replace('{', '').replace('}', '').replace('^', '**')], s))
    values = []
    st = 0
    while st < 1:
        for i in s:
            x.append(eval(i[0].replace('t', str(st))))
            y.append(eval(i[1].replace('t', str(st))) * (-1))
        st += 0.01
    fig, ax = plt.subplots()
    plt.axis('off')
    ax.scatter(x, y, s=50,color="black")
    fig.set_size_inches((max(x)-min(x))/10,4.8)
    fig.savefig("2.png")
    plt.close(fig)
    # plt.close(fig)
    return letters_extract("2.png")


def rec2(xx):
    x = []
    y = []
    s = bz2.decompress(base64.b85decode(xx)).decode("utf-8").strip().split('\n')

    s = s[1:]
    s = list(map(lambda x: x[8:-1], s))
    s = list(map(lambda x: [x.split(';')[0].replace('{', '').replace('}', '').replace('^', '**'),
                            x.split(';')[1].replace('{', '').replace('}', '').replace('^', '**')], s))
    values = []
    st = 0
    while st < 1:
        for i in s:
            x.append(eval(i[0].replace('t', str(st))))
            y.append(eval(i[1].replace('t', str(st))) * (-1))
        st += 0.01
    fig, ax = plt.subplots()
    plt.axis('off')
    ax.scatter(x, y, s=50, color="black")
    fig.set_size_inches((max(x) - min(x)) / 10, 4.8)
    fig.savefig("2.png")
    plt.show()
    sss=input()
    letters_extract2("2.png",sss)
    return sss


while True:
    depth=0
    flag=False
    soc=socket.socket()
    soc.connect(("tasks.aeroctf.com",40001))
    soc.recv(2048).decode("utf-8")
    soc.send(b'Y\n')
    ans=""
    while True:
        st = ""
        while not 'Result:' in st:
            st+=soc.recv(2048).decode("utf-8")
            if('Aero{' in st) and ('}' in st):
                print(st)
                exit(0)
            if 'Incorrect' in st:
                flag=True
                break

        if flag:
            print(depth)
            break


        st=st.strip().split('\n')[0].strip()
        ans=recon(st)
        print(ans)
        depth+=1
        soc.send((ans.strip()+'\n').encode('utf-8'))
