import numpy as np
import cv2
import matplotlib.pyplot as plt


def get_dominant_colors(img):
    # print(type(img))
    # height, width, channels = img.shape
    # print(height, width, channels)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = img_bgr[:, :, [2, 1, 0]]
    pixels = np.float32(img.reshape(-1, 3))
    n_colors = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    return palette


def main():
    img = cv2.imread('input/image.png')
    colours = get_dominant_colors(img)
    print(colours)
    exit(0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    average = img.mean(axis=0).mean(axis=0)
    print('average=', average)

    pixels = np.float32(img.reshape(-1, 3))

    n_colors = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS

    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)

    print('labels=', labels)
    print('palette=', palette)
    print('counts=', counts)

    dominant = palette[np.argmax(counts)]
    print('dominant=', dominant)

    avg_patch = np.ones(shape=img.shape, dtype=np.uint8) * np.uint8(average)

    indices = np.argsort(counts)[::-1]
    freqs = np.cumsum(np.hstack([[0], counts[indices] / float(counts.sum())]))
    rows = np.int_(img.shape[0] * freqs)

    dom_patch = np.zeros(shape=img.shape, dtype=np.uint8)
    for i in range(len(rows) - 1):
        dom_patch[rows[i]:rows[i + 1], :, :] += np.uint8(palette[indices[i]])

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6))
    ax0.imshow(avg_patch)
    ax0.set_title('Average color')
    ax0.axis('off')
    ax1.imshow(dom_patch)
    ax1.set_title('Dominant colors')
    ax1.axis('off')
    plt.show()

if __name__ == '__main__':
    main()