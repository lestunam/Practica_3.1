import numpy as np
import cv2

def mult_matrices(A, *matrices):
    result = np.array(A)
    for matrix in matrices:
        # Multiplicar usando np.dot
        result = np.dot(result, matrix.data)
    return result

def minimos_cuadrados(A, b):
    A = np.array(A)
    b = np.array(b)
    At = A.transpose()
    AtA = mult_matrices(At, A)
    AtAinv = np.linalg.inv(AtA)
    return mult_matrices(AtAinv, At,b)

def t_m(proy, ort, num_imgs):
    A = []
    b = []

    for i in range(num_imgs):
        x, y = proy[i]
        u, v = ort[i]

        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y])

        b.append([u])
        b.append([v])

    A = np.array(A)
    b = np.array(b)

    #U, S, Vh = np.linalg.svd(A)
    #t_matrix = Vh[-1].reshape(3, 3)
    #t_matrix = t_matrix / t_matrix[-1, -1]

    t_matrix = minimos_cuadrados(A, b)
    t_matrix = np.append(t_matrix,[1])

    return t_matrix

if __name__ == "__main__":

    num_imgs = 4

    p_p = [
        [114, 215],
        [302, 312],
        [244, 124],
        [375, 171]
    ]
    p_o = [
        [107, 102],
        [104, 388],
        [397, 103],
        [397, 340]
    ]

    t = t_m(p_p, p_o, num_imgs)
    t_r = t.reshape(3, 3)
    print(t_r)

    img_pro = cv2.imread('imagen_proyectada_sin_marco.png')

    height, width = img_pro.shape[:2]
    img_corregida = cv2.warpPerspective(img_pro, t_r, (width, height))

    cv2.imshow('Imagen Corregida', img_corregida)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("minimos_cuadrados.png",img_corregida)

