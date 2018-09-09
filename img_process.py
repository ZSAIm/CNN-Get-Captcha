

def binary(imgsrc, imgpx, threshold):
    for i in range(0, imgsrc.height):
        for j in range(0, imgsrc.width):
            if imgpx[j, i] > threshold:
                imgpx[j, i] = 255
            else:
                imgpx[j, i] = 0


def clear_noise(imgsrc, imgpx):
    for i in range(2):
        for y in range(0, imgsrc.height):
            for x in range(0, imgsrc.width):
                if imgpx[x, y] == 0:
                    if x - 1 < 0 or y - 1 < 0 or x + 1 >= imgsrc.width or y + 1 >= imgsrc.height:
                        imgpx[x, y] = 255
                        continue
                    count = 0
                    if imgpx[x + 1, y] == 255:
                        count += 1
                    if imgpx[x - 1, y] == 255:
                        count += 1
                    if imgpx[x, y + 1] == 255:
                        count += 1
                    if imgpx[x, y - 1] == 255:
                        count += 1
                    if count >= 3:
                        imgpx[x, y] = 255



if __name__ == '__main__':
    pass
    # main()



