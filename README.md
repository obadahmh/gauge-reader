# gauge-reader

Reads analog gauges from a robot camera and publishes the values over MQTT.

```
frame -> AprilTag -> front-on dial crop -> keypoint model (needle base, tip) -> angle -> value -> MQTT
```

The tag tells us which gauge it is and how to warp the dial flat. A small YOLO pose model finds the needle. The angle to value mapping comes from a one-time calibration per gauge.

## Setup

On the Jetson, install PyTorch for your JetPack first ([Ultralytics Jetson guide](https://docs.ultralytics.com/guides/nvidia-jetson)), then:

```
pip install -r requirements.txt
```

## 1. Mark the gauges

Print AprilTags from the `tag36h11` family and stick one next to each gauge, flat and roughly in the same plane as the dial face. Each gauge needs its own tag id. Bigger tags placed closer to the dial give a better warp, since small errors at the tag corners grow with distance.

## 2. Calibrate each gauge

Take a photo of the gauge and its tag, then:

```
python calibrate.py photo.jpg --name P-101 --unit bar
```

Click the dial center, a point on the rim, then two or more scale marks going clockwise. Press Enter and type the value of each mark. Paste the printed YAML under `gauges:` in `config.yaml`. Use more than two marks if the scale is not linear.

## 3. Train the needle model

Download [SyncG](https://huggingface.co/datasets/YihengDeng/syncG) (20k synthetic gauges, CC BY 4.0) and convert it:

```
python syncg_to_yolo.py path/to/syncG datasets/gauges
yolo pose train data=data.yaml model=yolo26n-pose.pt imgsz=320 epochs=100
```

Then add real data. Save crops from your own robot runs, label two keypoints (needle base, needle tip) and a box covering the whole crop in CVAT or Label Studio, export in YOLO pose format into `datasets/gauges`, and fine-tune:

```
python gauge.py --dump crops/                  # or: --image *.jpg
yolo pose train data=data.yaml model=runs/pose/train/weights/best.pt imgsz=320 epochs=50
```

Export the final `best.pt` to TensorRT on the Jetson and point `model` in `config.yaml` at the `.engine` file:

```
yolo export model=path/to/best.pt format=engine half=True imgsz=320
```

## 4. Run

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

## Before trusting it

Collect a few hundred robot captures with manually read values, from gauges and days the model was not trained on, and measure the error as a percentage of full scale.

## License note

Ultralytics is AGPL-3.0. Closed-source or commercial use needs an Ultralytics enterprise license, or swap the model for a permissively licensed one (for example RTMPose from MMPose, Apache-2.0). Only `Needle` in `gauge.py` would change.
