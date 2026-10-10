# gauge-reader

Reads analog gauges from a robot camera and publishes the values over MQTT.

```
frame -> AprilTag -> front-on dial crop -> keypoint model (needle base, tip) -> angle -> value -> MQTT
```

The AprilTag identifies the gauge and gives the homography to rectify the dial. A YOLO pose model locates the needle. A one-time calibration per gauge maps the needle angle to a value.

[vlm/](vlm/README.md) compares this pipeline with an end-to-end vision-language model.

## Setup

```
conda create -y -n gauge python=3.12 && conda activate gauge
pip install torch torchvision          # see the notes below
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"
```

- If the output is `False` with a driver warning, install the PyTorch build for your driver. For CUDA 12.x, use `--index-url https://download.pytorch.org/whl/cu126`.
- On the Jetson, install PyTorch as in the [Ultralytics Jetson guide](https://docs.ultralytics.com/guides/nvidia-jetson). To use the JetPack TensorRT in conda, run `echo /usr/lib/python3/dist-packages > "$(python -c 'import site; print(site.getsitepackages()[0])')/jetpack.pth"`.

## 1. Mark the gauges

Print `tag36h11` AprilTags with a white margin. Attach one tag next to each gauge, flat and approximately coplanar with the dial. Use a unique tag ID per gauge. Use large tags near the dial: corner errors increase with distance.

```
python -c "import cv2; d=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11); cv2.imwrite('tag7.png', cv2.aruco.generateImageMarker(d, 7, 800))"
```

## 2. Calibrate each gauge

Take a photo of the gauge and its tag. Then run (requires a display):

```
python calibrate.py photo.jpg --name P-101 --unit bar
```

Click the dial center, a point on the rim, and two or more scale marks in clockwise order. Press Enter and type the value of each mark. Paste the YAML output under `gauges:` in `config.yaml`. For a nonlinear scale, click more than two marks.

Angles are in degrees clockwise from 12 o'clock. Example: a 0 to 10 bar gauge with 0 at 7:30 and 10 at 4:30 gives `scale: [[225, 0], [135, 10]]`. Values are interpolated between marks. The calibration is valid while the tag does not move.

## 3. Train the needle model

Download and unpack [SyncG](https://huggingface.co/datasets/YihengDeng/syncG) (20k synthetic gauges, CC BY 4.0, approximately 24 GB):

```
hf download YihengDeng/syncG --repo-type dataset --include "syncG.*" --local-dir ~/data/syncG_raw
cd ~/data/syncG_raw && zip -s- syncG.zip -O syncG_full.zip && rm syncG.z0* syncG.zip && unzip -q syncG_full.zip && rm syncG_full.zip && cd -
```

Convert it, set its absolute path in `data.yaml` (Ultralytics resolves relative paths against its own datasets folder), and train:

```
python syncg_to_yolo.py ~/data/syncG_raw/syncG datasets/gauges
sed -i "s#^path: .*#path: $(pwd)/datasets/gauges#" data.yaml
yolo pose train data=data.yaml model=yolo26n-pose.pt imgsz=320 epochs=100 batch=64 device=0 workers=4
```

Each GPU process loads its own PyTorch and `workers` data loader processes. With limited RAM, decrease the GPU count and `workers`. To find OOM kills, use `journalctl | grep -i oom`. Per-epoch metrics are in `runs/pose/<run>/results.csv` and `results.png`.

Measure the needle angle error on the SyncG validation set:

```
python eval.py synth runs/pose/<run>/weights/best.pt
```

## 4. Test on your photos

Put the photos in a folder. Convert iPhone HEIC photos to JPEG:

```
python heic_to_jpg.py photos/
```

Calibrate each gauge (step 2). Then write the true value of each photo in `photos/truth.csv`:

```
file,value
IMG_0001.jpg,4.2
IMG_0002.jpg,6.8
```

If a photo shows more than one gauge, add a `tag` column. Then run:

```
python eval.py real runs/pose/<run>/weights/best.pt photos/
```

The script prints the error per photo as % of full scale and saves each rectified dial with the predicted needle to `crops/`. Typical gauge accuracy is 1 to 2% of full scale.

If accuracy on real photos is low, save their crops (`python gauge.py --dump crops/ --image photos/*.jpg`). In CVAT or Label Studio, label two keypoints (needle base, needle tip) and a box around the full crop. Export in YOLO pose format to `datasets/gauges` and fine-tune from `best.pt`.

## 5. Deploy on the Jetson

Copy `best.pt` to the Jetson and export it there. A TensorRT engine runs only on the GPU and TensorRT version that built it.

```
yolo export model=models/needle.pt format=engine half=True imgsz=320
```

`config.yaml` already points to `models/needle.engine`.

## 6. Run

```
python gauge.py                   # camera
python gauge.py --image *.jpg     # files
python gauge.py --crops crops/    # also save the crops, with a line on the needle
```

Run it when the robot stops at a gauge. Each run captures `frames` frames, discards blurred frames, and publishes the median per gauge to `<topic>/<gauge name>`:

```json
{"gauge": "P-101", "tag": 7, "value": 4.12, "unit": "bar", "n": 9, "spread": 0.03, "ts": 1791200000.0}
```

`spread` is the standard deviation across frames. A high `spread` indicates an unreliable reading. Each reading is also appended to `readings.jsonl`, so no reading is lost if the MQTT broker is unavailable.

## License note

Ultralytics is AGPL-3.0, which is acceptable for internal research. Closed-source or commercial use requires an Ultralytics enterprise license, or a permissively licensed model such as RTMPose from MMPose (Apache-2.0). Only the `Needle` class in `gauge.py` changes.
