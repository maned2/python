import json, cv2
import numpy as np
from progress.bar import IncrementalBar

COLOR = (0, 255, 0)
THICKNESS = 1  # in pixel
PROBABILITY_FILTER = 0.6

def draw_on_frame(frame, pt1, pt2, text):
    result = cv2.rectangle(
        frame,
        pt1,
        pt2,
        COLOR,
        THICKNESS
    )
    result = cv2.putText(
        result,
        text,
        pt1,
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        COLOR,
        THICKNESS,
        cv2.LINE_AA
    )

    return result

def open_file(fileName):
    f = open('input/' + fileName)
    data = json.load(f)
    f.close()
    return data


def main():



    data = open_file('pl_bboxes.json')
    print(data[0])

    cap = cv2.VideoCapture('input/video.mp4')
    frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    bar = IncrementalBar('Frames', max=frames_count)
    frame_num = 0
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output/video.avi', fourcc, 30.0, (5500, 500))

    while True:
        print(frame_num)
        ret, frame = cap.read()
        if not ret:
            break

        try:
            bboxes = data[frame_num]
            bbox_num = 0
            for j in bboxes:
                point_1 = (int(j[0]), int(j[1]))
                point_2 = (int(j[2]), int(j[3]))
                probability = j[4]
                if (probability > PROBABILITY_FILTER):
                    # TODO: crop image, detect avg color
                    frame = draw_on_frame(frame, point_1, point_2, str(bbox_num))
                    bbox_num += 1
        except Exception as e:
            print(e)
            break

        frame_num += 1
        bar.next()
        # cv2.imshow('video feed', frame)
        out.write(frame)

    bar.finish()
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
