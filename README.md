# gauge-reader

This software reads analog gauges from a robot camera. It sends the values over MQTT.

```
frame -> AprilTag -> front-on dial crop -> keypoint model (needle base, tip) -> angle -> value -> MQTT
```

The AprilTag identifies the gauge. The tag also gives the warp that makes the dial flat. A small YOLO pose model finds the needle. A calibration changes the needle angle into a value. You do the calibration one time for each gauge.

The [vlm/](vlm/) folder has a different method. A vision-language model reads the full gauge. See [vlm/README.md](vlm/README.md).

## Setup

```
conda create -y -n gauge python=3.12 && conda activate gauge
pip install torch torchvision          # see the notes below
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"
```

- If the last command shows `False` and a warning about an old driver, install the PyTorch build for your driver. For a CUDA 12.x driver, use `--index-url https://download.pytorch.org/whl/cu126`.
- On the Jetson, install PyTorch with the [Ultralytics Jetson guide](https://docs.ultralytics.com/guides/nvidia-jetson). To use the JetPack TensorRT in conda, run `echo /usr/lib/python3/dist-packages > "$(python -c 'import site; print(site.getsitepackages()[0])')/jetpack.pth"`.

## 1. Put tags on the gauges

Print AprilTags from the `tag36h11` family. Keep a white margin around each tag. Attach one tag adjacent to each gauge. Make sure that the tag is flat and approximately in the same plane as the dial. Each gauge must have a different tag id. Use large tags, and put them near the dial. Small errors at the tag corners become larger with distance.

```
python -c "import cv2; d=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11); cv2.imwrite('tag7.png', cv2.aruco.generateImageMarker(d, 7, 800))"
```

## 2. Calibrate each gauge

Take a photo of the gauge and its tag. Then run this command. It needs a screen.

```
python calibrate.py photo.jpg --name P-101 --unit bar
```

Click the dial center. Then click one point on the rim. Then click two or more scale marks in the clockwise direction. Push Enter, and type the value of each mark. Copy the YAML output into `config.yaml`, below `gauges:`. If the scale is not linear, click more than two marks.

The angles are in degrees, clockwise from 12 o'clock. Example: a 0 to 10 bar gauge has 0 at 7:30 and 10 at 4:30. The result is `scale: [[225, 0], [135, 10]]`. The software interpolates the value between the marks. The calibration stays correct if the tag does not move.

## 3. Train the needle model

Download and unpack [SyncG](https://huggingface.co/datasets/YihengDeng/syncG). It has 20k synthetic gauges (CC BY 4.0, approximately 24 GB).

```
hf download YihengDeng/syncG --repo-type dataset --include "syncG.*" --local-dir ~/data/syncG_raw
cd ~/data/syncG_raw && zip -s- syncG.zip -O syncG_full.zip && rm syncG.z0* syncG.zip && unzip -q syncG_full.zip && rm syncG_full.zip && cd -
```

Convert the dataset. Put the path of the result in `data.yaml` (Ultralytics finds relative paths from its datasets folder). Then train the model:

```
python syncg_to_yolo.py ~/data/syncG_raw/syncG datasets/gauges
sed -i "s#^path: .*#path: $(pwd)/datasets/gauges#" data.yaml
yolo pose train data=data.yaml model=yolo26n-pose.pt imgsz=320 epochs=100 batch=64 device=0 workers=4
```

Each GPU process loads PyTorch and `workers` loader processes. On a machine with little RAM, use few GPUs and a low `workers` value. To find out-of-memory kills, use `journalctl | grep -i oom`. The metrics for each epoch are in `runs/pose/<run>/results.csv` and `results.png`.

Measure the needle angle error on the SyncG validation set:

```
python eval.py synth runs/pose/<run>/weights/best.pt
```

## 4. Test on your photos

Put the photos in a folder. Convert iPhone HEIC photos first:

```
python heic_to_jpg.py photos/
```

Calibrate each gauge (step 2). Then write `photos/truth.csv`. For each photo, write the value that you read on the gauge:

```
file,value
IMG_0001.jpg,4.2
IMG_0002.jpg,6.8
```

If a photo shows more than one gauge, add a `tag` column. Then run:

```
python eval.py real runs/pose/<run>/weights/best.pt photos/
```

The script shows the error for each photo as a percentage of full scale. It saves each flat dial with the predicted needle in `crops/`. The accuracy of most gauges is 1 to 2% of full scale.

If the results on real photos are bad, save crops from them (`python gauge.py --dump crops/ --image photos/*.jpg`). In CVAT or Label Studio, label two keypoints (needle base, needle tip) and one box around the full crop. Export the labels in YOLO pose format into `datasets/gauges`. Then fine-tune from `best.pt`.

## 5. Install on the Jetson

Copy `best.pt` to the Jetson and export it on the Jetson. A TensorRT engine runs only on the GPU and the TensorRT version that made it.

```
yolo export model=models/needle.pt format=engine half=True imgsz=320
```

`config.yaml` already uses `models/needle.engine`.

## 6. Run

```
python gauge.py                   # camera
python gauge.py --image *.jpg     # files
python gauge.py --crops crops/    # also save the crops, with a line on the needle
```

Run the software when the robot stops at a gauge. Each run gets `frames` frames and ignores blurred frames. It sends the median value of each gauge to `<topic>/<gauge name>`:

```json
{"gauge": "P-101", "tag": 7, "value": 4.12, "unit": "bar", "n": 9, "spread": 0.03, "ts": 1791200000.0}
```

`spread` is the standard deviation of the frames. A high `spread` value shows that the reading is not reliable. The software also adds each reading to `readings.jsonl`. Thus no reading is lost if the MQTT broker is not available.

## License note

Ultralytics has the AGPL-3.0 license. This license is satisfactory for internal research. For closed-source or commercial use, you must have an Ultralytics enterprise license. Alternatively, use a model with a permissive license, for example RTMPose from MMPose (Apache-2.0). Only the `Needle` class in `gauge.py` must change.
