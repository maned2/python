import copy
import json, traceback, time
import cv2
import numpy as np
from progress.bar import IncrementalBar
from dominant_color import get_dominant_color

COLOR = (0, 255, 0)
THICKNESS = 1  # in pixel
PROBABILITY_FILTER = 0.6
CROP_PERCENT = 60
COLORS = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "magenta": (255, 0, 255),
    "cyan": (0, 255, 255),
    "black": (0, 0, 0),
    "white": (255, 255, 255)
}


def to_tuple(a):
    try:
        return tuple(to_tuple(i) for i in a)
    except TypeError:
        return a


def get_color_name(rgb):
    # print('=============get_color_name')
    min_distance = float("inf")
    closest_color = None
    for color, value in COLORS.items():
        distance = sum([(i - j) ** 2 for i, j in zip(rgb, value)])
        # print(distance, color)
        if distance < min_distance:
            min_distance = distance
            closest_color = color
    return closest_color


def draw_on_frame(frame, pt1, pt2, text, color_rgb):
    # color_rgb = to_tuple(color_rgb_float.astype(int))
    color_bgr = (
        color_rgb[2],
        color_rgb[1],
        color_rgb[0]
    )

    # print(color_rgb_float, color_rgb, color_bgr)
    # print(type(color_rgb_float), type(color_rgb), type(color_bgr))
    result = cv2.rectangle(
        frame,
        pt1,
        pt2,
        color_bgr,
        THICKNESS
    )
    result = cv2.putText(
        result,
        text,
        pt1,
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color_bgr,
        THICKNESS,
        cv2.LINE_AA
    )

    return result

def crop_top_part(img, bbox_num): # crop half
    # crop half
    height, width, channels = img.shape
    # print(height, width, channels)
    new_height = int(height * CROP_PERCENT / 100)
    cropped_img = img[0:new_height, 0:width]
    # cv2.imwrite('output/' + str(bbox_num) + '.jpg', img)
    # cv2.imwrite('output/' + str(bbox_num) + '_c.jpg', cropped_img)
    return cropped_img

def open_file(fileName):
    f = open('input/' + fileName)
    data = json.load(f)
    f.close()
    return data


def write_team(data, teams):
    result = copy.deepcopy(data)
    for i, frame in enumerate(result):
        for j, bbox in enumerate(frame):
            if len(result[i][j]) < 6:
                continue

            color_name = result[i][j][5]
            if color_name == teams[0]:
                result[i][j][5] = 0
            elif color_name == teams[1]:
                result[i][j][5] = 1
            else:
                result[i][j][5] = None

    return result

def test_write_team():
    data = open_file('result_intermediate.json')
    teams = ['black', 'red']
    data_result = write_team(data, teams)
    # print(data_result[0][0])
    # print(data_result[0][1])

    with open('output/result.json', 'w', encoding='utf-8') as f:
        json.dump(data_result, f, ensure_ascii=False, indent=4)


def main():
    colors_stat = {}
    data = open_file('pl_bboxes.json')
    # print(data[0])

    cap = cv2.VideoCapture('input/video.mp4')
    frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    bar = IncrementalBar('Frames', max=frames_count)
    frame_num = 0
    print(frames_count)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output/video.avi', fourcc, 30.0, (5500, 500))

    while True:
        # print(frame_num)
        ret, frame = cap.read()
        if not ret:
            break

        try:
            bboxes = data[frame_num]
            bbox_num = 0
            for j in bboxes:
                x1 = int(j[0])
                y1 = int(j[1])
                x2 = int(j[2])
                y2 = int(j[3])
                # width = x2 - x1
                # height = y2 - y1
                point_1 = (x1, y1)
                point_2 = (x2, y2)
                # print(x1, y1, x2, y2, width, height, point_1, point_2)
                probability = j[4]
                if probability > PROBABILITY_FILTER:
                    img = frame[y1:y2, x1:x2]
                    # print(img.size, width, height)
                    if img.size == 0:
                        continue

                    top_part = crop_top_part(img, bbox_num)  # TODO speed обрезать дважды - убрать
                    # k-mean color
                    color_rgb_float = get_dominant_color(top_part)
                    color_rgb = (
                        int(color_rgb_float[0]),
                        int(color_rgb_float[1]),
                        int(color_rgb_float[2])
                    )
                    # average color
                    # color_rgb_float_avg = top_part.mean(axis=0).mean(axis=0)
                    # color_rgb = (
                    #     int(color_rgb_float_avg[0]),
                    #     int(color_rgb_float_avg[1]),
                    #     int(color_rgb_float_avg[2])
                    # )
                    color_name = get_color_name(color_rgb)
                    data[frame_num][bbox_num].append(color_name)
                    data[frame_num][bbox_num].append(color_rgb)

                    if color_name in colors_stat:
                        colors_stat[color_name] += 1
                    else:
                        colors_stat[color_name] = 1
                    frame = draw_on_frame(frame, point_1, point_2, color_name, color_rgb)
                    bbox_num += 1

                    # file_name_tmp = "output/{}_{}_{}_{}_{}.jpg".format(bbox_num, color_name, color_rgb[0], color_rgb[1], color_rgb[2])
                    # cv2.imwrite(
                    #     file_name_tmp, top_part
                    # )
                    # if bbox_num == 3:
                    #     exit(0)
        except Exception as e:
            print(e)
            traceback.print_exc()
            break

        frame_num += 1
        bar.next()
        # cv2.imshow('video feed', frame)
        out.write(frame)

    bar.finish()
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(colors_stat)
    # max_value_1 = max(colors_stat.values())
    max_key_1 = max(colors_stat, key=colors_stat.get)
    colors_stat.pop(max_key_1, None)
    # max_value_2 = max(colors_stat.values())
    max_key_2 = max(colors_stat, key=colors_stat.get)
    teams = [max_key_1, max_key_2]
    print(teams)

    with open('output/result_intermediate.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    data_result = write_team(data, teams)
    print(data_result[0][0])
    print(data_result[0][1])

    with open('output/result.json', 'w', encoding='utf-8') as f:
        json.dump(data_result, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    start_time = time.time()
    main()
    # test_write_team()
    print('Execution Time: %s seconds' % (time.time() - start_time))
