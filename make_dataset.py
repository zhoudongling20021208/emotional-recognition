import os
import cv2 as cv
import numpy as np

make = True
check = True
if __name__ == "__main__":
    if make:
        all_data = []
        all_label = []
        numpy_data = []
        numpy_label = []
        numpy_flag = 0
        for i in range(10):
            path = './picture/%d'%i  #path会分别等于./picture/1  ./picture/2   ./picture/3    ....   ./picture/9
            for f in os.listdir(path):
                extension = os.path.splitext(f)[-1]
                if ( extension == '.jpg'):
                    img = cv.imread(os.path.join(path, f))
                    try:
                        img = cv.resize(img, (32,32))[...,(2,1,0)] # opencv read as bgr, but we need rgb
                        all_data.append(img)
                        all_label.append(i)
                    except:
                        continue
                if (extension == '.npy'):
                    npy_file = os.path.join(path, f)
                    tmp = np.load(npy_file)
                    numpy_data.append(tmp)
                    numpy_label += [i] * (len(tmp))
                    numpy_flag = 1
        if 1==numpy_flag:
            npy_tmp = numpy_data[0]
            for npy in numpy_data[1:]:
                npy_tmp = np.vstack([npy_tmp, npy])

            all_data = np.asarray(all_data)
            all_data = np.vstack([all_data, npy_tmp]) if len(all_data) else npy_tmp

            all_label = np.asarray(all_label + numpy_label)
            all_label = np.asarray(all_label)
        else:
            all_data = np.asarray(all_data)
            all_label = np.asarray(all_label)

        np.save("x", all_data)
        np.save("y", all_label)
    if check:
        x = np.load("x.npy")
        y = np.load("y.npy")
        label = ["0", "1", "2", "3", "4"] + ["5", "6", "7", "8", "9"]
        for d,idx in zip(x, y):
            print("Class %s"%label[idx])
            d = cv.resize(d, (32,32))[...,(2,1,0)]
            cv.imshow("img", d)
            cv.waitKey(1)
