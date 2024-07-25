# BirdNET-XRay

This is an interactive application that demonstrates how the BirdNET bird sound identification model works. This demo provides a visual and auditory experience where users can see and hear how the model processes and identifies bird calls in real time.

<img src="BirdNET-XRay_Demo.gif" alt="BirdNET-XRay Demo" style="width:100%;">

## Setup
```
sudo apt-get install ffmpeg portaudio19-dev python3-opencv
pip3 install tensorflow screeninfo librosa pyaudio
```

## Run
```
python3 demo.py
```

Command line arguments:

- `--resolution`: 
  - **Type**: `str`
  - **Default**: `'fullscreen'`
  - **Description**: Resolution of the window, e.g., `"fullscreen"` or `"1024x768"`.

- `--scaling`: 
  - **Type**: `float`
  - **Default**: `1.5`
  - **Description**: Scaling factor for the width of the output elements. Default is `1.5`, lower values might work better on smaller screens.

- `--fontsize`: 
  - **Type**: `float`
  - **Default**: `0.55`
  - **Description**: Font size for text elements. Default is `0.55`.

Keyboard shortcuts: 

- `esc` to quit
- `p` to pause/resume
- `s` to save the current frame
- `c` to change the colormap
- `a` to play next soundscape

## License

Feel free to use this demo for exhibitions or other projects.

If you do, please cite as:

```
@article{kahl2021birdnet,
  title={BirdNET: A deep learning solution for avian diversity monitoring},
  author={Kahl, Stefan and Wood, Connor M and Eibl, Maximilian and Klinck, Holger},
  journal={Ecological Informatics},
  volume={61},
  pages={101236},
  year={2021},
  publisher={Elsevier}
}
```

This work is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/deed).

## Funding

This project is supported by Jake Holshuh (Cornell class of '69) and The Arthur Vining Davis Foundations. Our work in the K. Lisa Yang Center for Conservation Bioacoustics is made possible by the generosity of K. Lisa Yang to advance innovative conservation technologies to inspire and inform the conservation of wildlife and habitats.

The German Federal Ministry of Education and Research is funding the development of BirdNET through the project "BirdNET+" (FKZ 01|S22072).
Additionally, the German Federal Ministry of Environment, Nature Conservation and Nuclear Safety is funding the development of BirdNET through the project "DeepBirdDetect" (FKZ 67KI31040E).

## Partners

BirdNET is a joint effort of partners from academia and industry.
Without these partnerships, this project would not have been possible.
Thank you!

![Our partners](https://tuc.cloud/index.php/s/KSdWfX5CnSRpRgQ/download/box_logos.png)