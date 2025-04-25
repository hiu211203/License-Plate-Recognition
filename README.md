# License Plate Recognition System

This project implements a real-time vehicle license plate recognition system using YOLOv5, OpenCV, and PyTorch. It encompasses license plate detection, segmentation, and character recognition.

---

## License Plate Detection Dataset

This repo uses 2 sets of data for 2 stage of license plate recognition problem:
- [Character Detection Dataset](https://drive.google.com/file/d/1xchPXf7a1r466ngow_W_9bittRqQEf_T/view?usp=sharing)
- [Character Detection Dataset](https://drive.google.com/file/d/1bPux9J0e1mz-_Jssx4XX1-wPGamaS8mI/view?usp=sharing)
  
---

## Features

- **License Plate Detection**: Trained a YOLOv5 model to detect license plates in diverse environments.
- **Character Recognition**: Used another YOLOv5 model to detect and classify individual characters from the plates.
- **Web Interface**: Simple Flask web app for image upload and result display.
  
---

## Installation

1. **Clone the repository**
   
```bash
git clone https://github.com/hiu211203/License-Plate-Recognition.git
cd License-Plate-Recognition
```

2. **Create and activate a virtual environment**
```bash
python -m venv venv
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirement.txt
```

4. **Run the app**
```bash
python app.py
```

---

## Contact

For any questions or feedback, please reach out to [hieu211203@gmail.com](mailto:hieu211203@gmail.com).

---
