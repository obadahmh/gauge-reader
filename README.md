# gauge-reader

Reads analog gauges from a robot camera and publishes the values over MQTT.

```
frame -> AprilTag -> front-on dial crop -> keypoint model (needle base, tip) -> angle -> value -> MQTT
```

The tag tells us which gauge it is and how to warp the dial flat. A small YOLO pose model finds the needle. The angle to value mapping comes from a one-time calibration per gauge.

## Setup

```
conda create -y -n gauge python=3.12 && conda activate gauge
pip install torch torchvision          # see the notes below
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"
```

- If this prints `False` and warns that the driver is too old, install the PyTorch build that matches your driver, for example `--index-url https://download.pytorch.org/whl/cu126` for a CUDA 12.x driver.
- On the Jetson, follow the [Ultralytics Jetson guide](https://docs.ultralytics.com/guides/nvidia-jetson) for PyTorch. To use JetPack's TensorRT inside conda, run `echo /usr/lib/python3/dist-packages > "$(python -c 'import site; print(site.getsitepackages()[0])')/jetpack.pth"`.

## 1. Mark the gauges

Print AprilTags from the `tag36h11` family, with a white margin, and stick one next to each gauge, flat and roughly in the same plane as the dial face. Each gauge needs its own tag id. Bigger tags placed closer to the dial give a better warp, since small errors at the tag corners grow with distance.

```
python -c "import cv2; d=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11); cv2.imwrite('tag7.png', cv2.aruco.generateImageMarker(d, 7, 800))"
```

## 2. Calibrate each gauge

Take a photo of the gauge and its tag, then (needs a screen):

```
python calibrate.py photo.jpg --name P-101 --unit bar
```

Click the dial center, a point on the rim, then two or more scale marks going clockwise. Press Enter and type the value of each mark. Paste the printed YAML under `gauges:` in `config.yaml`. Use more than two marks if the scale is not linear.

Angles are in degrees clockwise from 12 o'clock. A 0 to 10 bar gauge with 0 at 7:30 and 10 at 4:30 comes out as `scale: [[225, 0], [135, 10]]`, and the value is interpolated between the marks. The calibration holds as long as the tag does not move.

## 3. Train the needle model

Download and unpack [SyncG](https://huggingface.co/datasets/YihengDeng/syncG) (20k synthetic gauges, CC BY 4.0, about 24 GB):

```
hf download YihengDeng/syncG --repo-type dataset --include "syncG.*" --local-dir ~/data/syncG_raw
cd ~/data/syncG_raw && zip -s- syncG.zip -O syncG_full.zip && rm syncG.z0* syncG.zip && unzip -q syncG_full.zip && rm syncG_full.zip && cd -
```

Convert it, point `data.yaml` at the result (Ultralytics resolves relative paths against its own datasets folder), and train:

```
python syncg_to_yolo.py ~/data/syncG_raw/syncG datasets/gauges
sed -i "s#^path: .*#path: $(pwd)/datasets/gauges#" data.yaml
yolo pose train data=data.yaml model=yolo26n-pose.pt imgsz=320 epochs=100 batch=64 device=0 workers=4
```

Each GPU process loads its own copy of PyTorch plus `workers` loader processes, so on machines with little RAM keep the GPU count and `workers` low. Out of memory kills show up in `journalctl | grep -i oom`. Metrics per epoch are in `runs/pose/<run>/results.csv` and `results.png`.

Check the needle angle error on the SyncG validation set:

```
python eval.py synth runs/pose/<run>/weights/best.pt
```

## 4. Test on your own photos

Put the photos in a folder. iPhone HEIC photos need converting first:

```
python heic_to_jpg.py photos/
```

Calibrate each gauge as in step 2, then write `photos/truth.csv` with the value you read off the gauge for each photo:

```
file,value
IMG_0001.jpg,4.2
IMG_0002.jpg,6.8
```

Add a `tag` column if a photo shows more than one gauge. Then:

```
python eval.py real runs/pose/<run>/weights/best.pt photos/
```

It prints the error per photo as a percentage of full scale and saves each flattened dial with the predicted needle to `crops/`. Gauges themselves are usually accurate to 1 to 2% of full scale.

If real photos read poorly, save crops from them (`python gauge.py --dump crops/ --image photos/*.jpg`), label two keypoints (needle base, needle tip) and a box covering the whole crop in CVAT or Label Studio, export in YOLO pose format into `datasets/gauges`, and fine-tune from `best.pt`.

## 5. Deploy on the Jetson

Copy `best.pt` over and export it there, since TensorRT engines only run on the GPU and TensorRT version they were built with:

```
yolo export model=models/needle.pt format=engine half=True imgsz=320
```

`config.yaml` already points at `models/needle.engine`.

## 6. Run

```
python gauge.py                   # camera
python gauge.py --image *.jpg     # files
python gauge.py --crops crops/    # also save crops with the detected needle drawn on
```

Call it when the robot stops at a gauge. Each run grabs `frames` frames, skips blurry ones, and publishes the median per gauge to `<topic>/<gauge name>`:

```json
{"gauge": "P-101", "tag": 7, "value": 4.12, "unit": "bar", "n": 9, "spread": 0.03, "ts": 1791200000.0}
```

`spread` is the standard deviation across frames. A high value means the reading is unreliable. Every reading is also appended to `readings.jsonl`, so nothing is lost if the broker is unreachable.

## License note

Ultralytics is AGPL-3.0, which is fine for internal research. Closed-source or commercial use needs an Ultralytics enterprise license, or swap the model for a permissively licensed one (for example RTMPose from MMPose, Apache-2.0). Only `Needle` in `gauge.py` would change.
