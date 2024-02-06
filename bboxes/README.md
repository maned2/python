# Task

для каждого релевантного ббокса получить номер команды 0 или 1, None если ббокс не релевантен
(опционально) по результатам видео получить цвет каждой из команд в ргб формате
В конце хочется посмотреть на рендер или любую другую иллюстрацию проделанной работы

# Installation
```bash
# copy files into input folder: 
ball_bboxes.json
pl_bboxes.json
ref_bboxes.json
gkeep_bboxes.json
video.mp4
RUN pip install -r requirements.txt
```

# Run
```bash
py main.py
docker-compose up
```

# DEV
```bash
docker build .
```