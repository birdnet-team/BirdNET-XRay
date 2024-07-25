import os
import argparse
import cv2
from screeninfo import get_monitors
import numpy as np
from tensorflow import lite as tflite
import pyaudio
from threading import Thread
import traceback as tb
import librosa
import time

MODEL_PATH = 'model/BirdNET_GLOBAL_6K_V2.4_Model_FP16.tflite'
LABEL_FILE = 'model/BirdNET_GLOBAL_6K_V2.4_Labels.txt'

AUDIO_INDEX = 0
AUDIO_FOLDER = 'soundscapes'
AUDIO_DATA = []
AUDIO_SAMPLES = np.array([], dtype='float32')
BUFFER_SIZE = 1024
IMAGE_CHANNELS = 1
COLORMAP = None
SPACING = 0.0225
BORDER_COLOR = 128
FONT_SIZE = 0.55
TEXT_COLOR = (255, 255, 255)
NUMBER_OF_RESULTS = 15
MAXIMA = {}
STREAM = False
PAUSE = False

OUTPUT_IDX = {'spec1': 220, 'spec2': 261, 'conv0': 266, 'block1': 294, 'block2': 370, 'block3': 465, 'block4': 522, 'post_conv': 544, 'pooling': 545, 'class': 546}
GRID_WIDTH = {'spec1': 1, 'spec2': 1, 'conv0': 2, 'block1': 2, 'block2': 2, 'block3': 2, 'block4': 3, 'post_conv': 6, 'pooling': 11, 'class': 30}
SCREEN_WIDTH = {'spec1': 0.2, 'spec2': 0.2, 'conv0': 0.2, 'block1': 0.125, 'block2': 0.1, 'block3': 0.1, 'block4': 0.1, 'post_conv': 0.1, 'pooling': 0.1, 'class': 0.2, 'bar_width': 0.05}

def load(frame_width, frame_height, width_scaling):

    global interpreter, input_details, output_details, LABELS, width, height, SCREEN_WIDTH

    # Calculate the sum of the current values
    total = sum(SCREEN_WIDTH.values())

    # Normalize the values
    SCREEN_WIDTH = {key: (value / total) * width_scaling for key, value in SCREEN_WIDTH.items()}

    # load audio files
    loadSoundfiles()

    # Load labels file
    LABELS = []
    with open(LABEL_FILE, 'r') as f:
        for line in f:
            label = line.strip().split('_')[1]
            label = label.replace('Ã¤', 'ä').replace('Ã¶', 'ö').replace('Ã¼', 'ü').replace('ÃŸ', 'ß')
            label = label.replace('ä', 'ae').replace('ö', 'oe').replace('ü', 'ue').replace('ß', 'ss')
            LABELS.append(label)

    # Load model
    interpreter = tflite.Interpreter(model_path=MODEL_PATH, experimental_preserve_all_tensors=True, num_threads=4)

    # Allocate tensors
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Create window
    cv2.namedWindow('demo', cv2.WINDOW_NORMAL)

    # Get screen resolution
    screen = get_monitors()[0]
    screen_width = screen.width
    screen_height = screen.height

    if frame_width == -1 and frame_height == -1:
        width = screen_width
        height = screen_height

        # Show image in window full screen
        cv2.setWindowProperty('demo', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.setWindowProperty('demo', height, width)

    else:
        # Ensure the window size does not exceed the screen size
        width = min(frame_width, screen_width)
        height = min(frame_height, screen_height)

        # Set window position
        cv2.moveWindow('demo', 0, 0)

        # Set window size
        cv2.resizeWindow('demo', width, height)

def loadSoundfiles():

    global AUDIO_DATA

    # Parse audio folder and look for mp3 files
    afiles = [os.path.join(AUDIO_FOLDER, f) for f in os.listdir(AUDIO_FOLDER) if f.endswith('.mp3')]

    # Load raw audio data for all files
    AUDIO_DATA = []
    for f in afiles:
        print("Loading audio file: {} ({}/{})".format(f, afiles.index(f) + 1, len(afiles)), flush=True)
        sig, rate = librosa.load(f, sr=48000, offset=0, duration=None)
        AUDIO_DATA.append(sig)


def record():

    global AUDIO_SAMPLES

    # Open microphone stream
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32, 
                    channels=1, 
                    rate=48000, 
                    input=True, 
                    input_device_index=1,
                    frames_per_buffer=BUFFER_SIZE)

    # Record audio
    while STREAM:
        if not PAUSE:
            data = stream.read(BUFFER_SIZE, exception_on_overflow=False)
            data = np.frombuffer(data, 'float32')
            AUDIO_SAMPLES = np.concatenate((AUDIO_SAMPLES, data))
            AUDIO_SAMPLES = AUDIO_SAMPLES[-144000:]

    # Close microphone stream
    stream.stop_stream()

def play():

    global AUDIO_SAMPLES, AUDIO_INDEX

    # Open audio stream
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=48000,
                    output=True)

    # Play audio
    idx = 0
    sig = AUDIO_DATA[AUDIO_INDEX]
    chunk = sig[idx:idx + BUFFER_SIZE]
    while STREAM:
        if not PAUSE:
            if len(chunk) == BUFFER_SIZE:
                stream.write(chunk.astype('float32').tobytes())
                AUDIO_SAMPLES = np.concatenate((AUDIO_SAMPLES, chunk))
                AUDIO_SAMPLES = AUDIO_SAMPLES[-144000:]
                idx += BUFFER_SIZE
                chunk = sig[idx:idx + BUFFER_SIZE]
            else:
                AUDIO_INDEX += 1
                sig = AUDIO_DATA[AUDIO_INDEX]
                idx = 0
                chunk = sig[idx:idx + BUFFER_SIZE]

    # Close audio stream
    stream.stop_stream()

# DEBUG: For each layer, show output details
"""
def listTensors(interpreter):

    # Create dummy input
    dummy_input = np.zeros((1, 144000), dtype=np.float32)

    # Run model
    interpreter.set_tensor(input_details[0]['index'], dummy_input)
    interpreter.invoke()

    # Show output shape for all tensors
    for i in range(len(interpreter.get_tensor_details())):
        data = interpreter.get_tensor(i)
        print(i, interpreter.get_tensor_details()[i]["name"], data.shape)        

listTensors(interpreter)
sys.exit()
"""

def plotWaveform(sig, height, width):

    # Create image
    img = np.zeros((height, width), np.uint8)

    # No negative values
    sig = np.abs(sig)

    # Resize signal to width
    sig = cv2.resize(sig, (width, 1), interpolation=cv2.INTER_AREA)

    # Avoid NaNs
    sig = np.nan_to_num(sig)

    # Normalize signal
    if not 'wf' in MAXIMA:
        MAXIMA['wf'] = [np.max(sig)]
    else:
        MAXIMA['wf'].append(np.max(sig))
        MAXIMA['wf'] = MAXIMA['wf'][-100:]
    #sig = (sig - np.min(sig)) / ((np.max(sig) - np.min(sig)) + 0.000001)    
    sig = (sig - np.min(sig)) / ((np.max(MAXIMA['wf']) - np.min(sig)) + 0.000001)
    sig = sig[0] 

    # Plot signal
    for i in range(sig.shape[0]):
        img[int((1 - sig[i]) / 2 * img.shape[0]):int((1 - sig[i]) / 2 * img.shape[0] + sig[i] * img.shape[0]), i] = 255

    return img

def parseOutput(output_data, grid_width, frame_width, frame_height, name='', border_width=1, show_cell_border=True, border_color=128, frame_border_color=128, normalize=True, apply_relu=False, apply_sigmoid=False, min_value=0, threshold=0):

    # Determine grid height
    grid_height = int(np.ceil(output_data.shape[-1] / grid_width))

    # Apply relu
    if apply_relu:
        output_data = np.maximum(output_data, 0)

    # Apply sigmoid
    if apply_sigmoid:
        output_data = 1 / (1 + np.exp(-output_data))

    # Normalize output
    if normalize:
        if not name in MAXIMA:
            MAXIMA[name] = [np.max(output_data)]
        else:
            MAXIMA[name].append(np.max(output_data))
            MAXIMA[name] = MAXIMA[name][-25:]
        output_data = np.clip((output_data - np.min(output_data)) / (np.max(output_data) * 0.75 - np.min(output_data) + 0.000001), 0, 1) * 255
        #output_data = np.minimum(1, (output_data - np.min(output_data)) / (np.max(MAXIMA[name]) * 0.7 - np.min(output_data) + 0.00000001)) * 255
    else:
        output_data = output_data * 255

    # Apply threshold
    if threshold > 0:
        output_data[output_data < threshold] = 0

    # Set min value
    if min_value > 0:
        output_data[output_data < min_value] = min_value

    # Create dummy frame
    # Each grid cell is based on output shape
    # and has 1px white border
    cell_width = output_data.shape[2]
    cell_height = output_data.shape[1]
    frame = np.zeros((int(grid_height * cell_height), int(grid_width * cell_width), 1), np.uint8)

    # For each grid cell
    for i in range(output_data.shape[-1]):
        x = i % grid_width
        y = int(i / grid_width)

        # Add axxis to output
        output = np.expand_dims(output_data[0, :, :, i], axis=-1)

        # Put output in center of grid cell
        frame[y * cell_height:y * cell_height + output.shape[0], x * cell_width:x * cell_width + output.shape[1]] = output   

    # Resize frame to frame_width x frame_height and keep aspect ratio
    scale = frame_width / frame.shape[1]
    if frame.shape[0] * scale > frame_height:
        scale = frame_height / frame.shape[0]
    frame = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)), interpolation=cv2.INTER_NEAREST )       
        

    # Add border between grid cells
    if border_width > 0:

        if show_cell_border:
            # Vertical borders
            for i in range(grid_width - 1):
                frame[:, int((i + 1) * cell_width * scale + border_width):int((i + 1) * cell_width * scale + border_width * 2)] = border_color

            # Horizontal borders
            for i in range(grid_height - 1):
                frame[int((i + 1) * cell_height * scale + border_width):int((i + 1) * cell_height * scale + border_width * 2), :] = border_color

        # Border around entire frame
        frame = cv2.copyMakeBorder(frame, border_width, border_width, border_width, border_width, cv2.BORDER_CONSTANT, value=frame_border_color)
    
    # Expand dims
    frame = np.expand_dims(frame, axis=-1)

    # Convert into RGB by repeating 3 times
    #if IMAGE_CHANNELS > 1:
    #    frame = np.repeat(frame, IMAGE_CHANNELS, axis=-1)

    # Convert into RGB by applying viridis colormap
    if IMAGE_CHANNELS > 1 and COLORMAP is not None:
        frame = cv2.applyColorMap(frame, COLORMAP)

    return frame	    

def main():

    global IMAGE_CHANNELS, TEXT_COLOR, COLORMAP, PAUSE

    # Loop until user press ESC
    while True:

        # Create black dummy frame at screen resolution
        frame = np.zeros((height, width, IMAGE_CHANNELS), np.uint8)

        # Read from stream        
        sig = AUDIO_SAMPLES.copy()

        # If signal is shorter than 144000 samples, pad with zeros
        if len(sig) < 144000:
            sig = np.pad(sig, (0, 144000 - len(sig)), 'constant')

        # Reshape signal to 1x144000
        sig = sig.reshape(1, 144000)

        # Run model
        #t_start = time.time()
        interpreter.set_tensor(input_details[0]['index'], sig)
        interpreter.invoke()
        #print("Inference time: {} ms".format((time.time() - t_start) * 1000))

        # Get output for spectrogram layer
        out_tensor = interpreter.get_tensor(OUTPUT_IDX['spec2'])
        out_tensor = np.expand_dims(out_tensor, axis=-1)
        output_spec = parseOutput(out_tensor, GRID_WIDTH['spec2'], int(width * SCREEN_WIDTH['spec2']), int(height * (0.33 - SPACING * 2)), name='spec2', border_width=1, border_color=BORDER_COLOR, frame_border_color=BORDER_COLOR)
        output_spec_posX = int(width * SPACING)
        output_spec_posY = int(height * 0.15)
        frame[output_spec_posY:output_spec_posY + output_spec.shape[0], output_spec_posX:output_spec_posX + output_spec.shape[1]] = output_spec
        spec_text = "Spectrogram, {}x{} pixel".format(out_tensor.shape[1], out_tensor.shape[2])
        cv2.putText(frame, spec_text, (output_spec_posX, output_spec_posY - 10), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, TEXT_COLOR, 1, cv2.LINE_AA)    

        # Get output for waveform layer
        waveform = plotWaveform(sig, out_tensor.shape[1], out_tensor.shape[2])   
        waveform = np.expand_dims(waveform, axis=0)  
        waveform = np.expand_dims(waveform, axis=-1)
        output_wave = parseOutput(waveform, 1, int(width * SCREEN_WIDTH['spec2']), int(height * (0.33 - SPACING * 2)), name='wave', border_width=1, border_color=BORDER_COLOR, frame_border_color=BORDER_COLOR)
        output_wave_posX = int(width * SPACING)
        output_wave_posY = int(height * SPACING)
        frame[output_wave_posY:output_wave_posY + output_wave.shape[0], output_wave_posX:output_wave_posX + output_wave.shape[1]] = output_wave
        wave_text = "Audio input stream, 3s @ 48kHz"
        cv2.putText(frame, wave_text, (output_wave_posX, output_wave_posY - 10), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, TEXT_COLOR, 1, cv2.LINE_AA)

        # Get output for conv0 layer
        out_tensor = interpreter.get_tensor(OUTPUT_IDX['conv0'])
        output_conv0 = parseOutput(out_tensor, GRID_WIDTH['conv0'], int(width * SCREEN_WIDTH['conv0']), int(height * (0.725 - SPACING * 2)), name='conv0', border_width=1, border_color=BORDER_COLOR, frame_border_color=BORDER_COLOR, apply_relu=False)
        #output_conv0 = cv2.resize(output_conv0, (output_spec.shape[1], output_conv0.shape[0]), interpolation=cv2.INTER_NEAREST)
        output_conv0 = cv2.resize(output_conv0, (output_spec.shape[1], int(height - SPACING * 2 - output_spec_posY - output_spec.shape[0] * 2)), interpolation=cv2.INTER_NEAREST)
        if len(output_conv0.shape) == 2:
            output_conv0 = np.expand_dims(output_conv0, axis=-1)
        output_conv0_posX = int(width * SPACING)
        output_conv0_posY = height - int(height * SPACING) - output_conv0.shape[0]
        frame[output_conv0_posY:output_conv0_posY + output_conv0.shape[0], output_conv0_posX:output_conv0_posX + output_conv0.shape[1]] = output_conv0
        conv0_text = "Pre-processing convolution, {} filers, {}x{} outputs".format(out_tensor.shape[-1], out_tensor.shape[1], out_tensor.shape[2])
        cv2.putText(frame, conv0_text, (output_conv0_posX, output_conv0_posY - 10), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, TEXT_COLOR, 1, cv2.LINE_AA)

        # Get output for block 1
        out_tensor = interpreter.get_tensor(OUTPUT_IDX['block1'])
        output_block1 = parseOutput(out_tensor, GRID_WIDTH['block1'], int(width * SCREEN_WIDTH['block1']), int(height * (1 - SPACING * 2)), name='block1', border_width=1, border_color=BORDER_COLOR, frame_border_color=BORDER_COLOR, apply_relu=True)
        output_block1_posX = output_spec_posX + output_spec.shape[1] + int(width * SPACING)
        output_block1_posY = int(height * SPACING)
        frame[output_block1_posY:output_block1_posY + output_block1.shape[0], output_block1_posX:output_block1_posX + output_block1.shape[1]] = output_block1

        # Add text on block 1
        text_frame = np.zeros((int(width * SPACING * 0.5), int(height * 0.5), IMAGE_CHANNELS), np.uint8)
        block_1_text = "Inverted ResBlock 1, {} filters, {}x{} outputs".format(out_tensor.shape[-1], out_tensor.shape[1], out_tensor.shape[2])
        cv2.putText(text_frame, block_1_text, (0, int(width * SPACING * 0.25)), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, TEXT_COLOR, 1, cv2.LINE_AA)
        text_frame = np.rot90(text_frame, 3)
        frame[output_block1_posY:output_block1_posY + text_frame.shape[0], output_block1_posX - int(width * SPACING * 0.5):output_block1_posX - int(width * SPACING * 0.5) + text_frame.shape[1]] = text_frame
    
        # Get output for block 2
        out_tensor = interpreter.get_tensor(OUTPUT_IDX['block2'])
        output_block2 = parseOutput(out_tensor, GRID_WIDTH['block2'], int(width * SCREEN_WIDTH['block2']), int(height * (1 - SPACING * 2)), name='block2', border_width=1, border_color=BORDER_COLOR, frame_border_color=BORDER_COLOR, apply_relu=True)
        output_block2_posX = output_block1_posX + output_block1.shape[1] + int(width * SPACING)
        output_block2_posY = int(height * SPACING)
        frame[output_block2_posY:output_block2_posY + output_block2.shape[0], output_block2_posX:output_block2_posX + output_block2.shape[1]] = output_block2

        # Add text on block 2
        text_frame = np.zeros((int(width * SPACING * 0.5), int(height * 0.5), IMAGE_CHANNELS), np.uint8)
        block_2_text = "Inverted ResBlock 2, {} filters, {}x{} outputs".format(out_tensor.shape[-1], out_tensor.shape[1], out_tensor.shape[2])
        cv2.putText(text_frame, block_2_text, (0, int(width * SPACING * 0.25)), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, TEXT_COLOR, 1, cv2.LINE_AA)
        text_frame = np.rot90(text_frame, 3)
        frame[output_block2_posY:output_block2_posY + text_frame.shape[0], output_block2_posX - int(width * SPACING * 0.5):output_block2_posX - int(width * SPACING * 0.5) + text_frame.shape[1]] = text_frame

        # Get output for block 3
        out_tensor = interpreter.get_tensor(OUTPUT_IDX['block3'])
        output_block3 = parseOutput(out_tensor, GRID_WIDTH['block3'], int(width * SCREEN_WIDTH['block3']), int(height * (1 - SPACING * 2)), name='block3', border_width=1, border_color=BORDER_COLOR, frame_border_color=BORDER_COLOR, apply_relu=True)   
        output_block3_posX = output_block2_posX + output_block2.shape[1] + int(width * SPACING)
        output_block3_posY = int(height * SPACING)
        frame[output_block3_posY:output_block3_posY + output_block3.shape[0], output_block3_posX:output_block3_posX + output_block3.shape[1]] = output_block3

        # Add text on block 3
        text_frame = np.zeros((int(width * SPACING * 0.5), int(height * 0.5), IMAGE_CHANNELS), np.uint8)
        block_3_text = "Inverted ResBlock 3, {} filters, {}x{} outputs".format(out_tensor.shape[-1], out_tensor.shape[1], out_tensor.shape[2])
        cv2.putText(text_frame, block_3_text, (0, int(width * SPACING * 0.25)), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, TEXT_COLOR, 1, cv2.LINE_AA)
        text_frame = np.rot90(text_frame, 3)
        frame[output_block3_posY:output_block3_posY + text_frame.shape[0], output_block3_posX - int(width * SPACING * 0.5):output_block3_posX - int(width * SPACING * 0.5) + text_frame.shape[1]] = text_frame

        # Get output for block 4
        out_tensor = interpreter.get_tensor(OUTPUT_IDX['block4'])
        output_block4 = parseOutput(out_tensor, GRID_WIDTH['block4'], int(width * SCREEN_WIDTH['block4']), int(height * (1 - SPACING * 2)), name='block4', border_width=1, border_color=BORDER_COLOR, frame_border_color=BORDER_COLOR, apply_relu=True)
        output_block4_posX = output_block3_posX + output_block3.shape[1] + int(width * SPACING)
        output_block4_posY = int(height * SPACING)
        frame[output_block4_posY:output_block4_posY + output_block4.shape[0], output_block4_posX:output_block4_posX + output_block4.shape[1]] = output_block4

        # Add text on block 4
        text_frame = np.zeros((int(width * SPACING * 0.5), int(height * 0.5), IMAGE_CHANNELS), np.uint8)
        block_4_text = "Inverted ResBlock 4, {} filters, {}x{} outputs".format(out_tensor.shape[-1], out_tensor.shape[1], out_tensor.shape[2])
        cv2.putText(text_frame, block_4_text, (0, int(width * SPACING * 0.25)), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, TEXT_COLOR, 1, cv2.LINE_AA)
        text_frame = np.rot90(text_frame, 3)
        frame[output_block4_posY:output_block4_posY + text_frame.shape[0], output_block4_posX - int(width * SPACING * 0.5):output_block4_posX - int(width * SPACING * 0.5) + text_frame.shape[1]] = text_frame
        
        # Get output for post conv
        out_tensor = interpreter.get_tensor(OUTPUT_IDX['post_conv'])
        output_post_conv = parseOutput(out_tensor, GRID_WIDTH['post_conv'], int(width * SCREEN_WIDTH['post_conv']), int(height * (1 - SPACING * 2)), name='post_conv', border_width=1, border_color=BORDER_COLOR, frame_border_color=BORDER_COLOR, show_cell_border=True, apply_relu=False)
        output_post_conv_posX = output_block4_posX + output_block4.shape[1] + int(width * SPACING)
        output_post_conv_posY = int(height * SPACING)
        frame[output_post_conv_posY:output_post_conv_posY + output_post_conv.shape[0], output_post_conv_posX:output_post_conv_posX + output_post_conv.shape[1]] = output_post_conv

        # Add text on post conv
        text_frame = np.zeros((int(width * SPACING * 0.5), int(height * 0.5), IMAGE_CHANNELS), np.uint8)
        post_conv_text = "Post-pocessing convolution, {} filters, {}x{} outputs".format(out_tensor.shape[-1], out_tensor.shape[1], out_tensor.shape[2])
        cv2.putText(text_frame, post_conv_text, (0, int(width * SPACING * 0.25)), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, TEXT_COLOR, 1, cv2.LINE_AA)
        text_frame = np.rot90(text_frame, 3)
        frame[output_post_conv_posY:output_post_conv_posY + text_frame.shape[0], output_post_conv_posX - int(width * SPACING * 0.5):output_post_conv_posX - int(width * SPACING * 0.5) + text_frame.shape[1]] = text_frame

        # Get output for pooling
        out_tensor = interpreter.get_tensor(OUTPUT_IDX['pooling'])
        out_tensor = np.expand_dims(out_tensor, axis=0)
        out_tensor = np.expand_dims(out_tensor, axis=0)
        output_pooling = parseOutput(out_tensor, GRID_WIDTH['pooling'], int(width * SCREEN_WIDTH['pooling']), int(height * (1 - SPACING * 2)), name='pooling', border_width=1, border_color=BORDER_COLOR, frame_border_color=BORDER_COLOR, show_cell_border=True, apply_relu=False, threshold=48)
        output_pooling_posX = output_post_conv_posX + output_post_conv.shape[1] + int(width * SPACING)
        output_pooling_posY = int(height * SPACING)
        frame[output_pooling_posY:output_pooling_posY + output_pooling.shape[0], output_pooling_posX:output_pooling_posX + output_pooling.shape[1]] = output_pooling

        # Add text on pooling
        text_frame = np.zeros((int(width * SPACING * 0.5), int(height * 0.5), IMAGE_CHANNELS), np.uint8)
        pooling_text = "Global average pooling, {}x{} outputs".format(out_tensor.shape[-1], out_tensor.shape[1], out_tensor.shape[2])
        cv2.putText(text_frame, pooling_text, (0, int(width * SPACING * 0.25)), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, TEXT_COLOR, 1, cv2.LINE_AA)
        text_frame = np.rot90(text_frame, 3)
        frame[output_pooling_posY:output_pooling_posY + text_frame.shape[0], output_pooling_posX - int(width * SPACING * 0.5):output_pooling_posX - int(width * SPACING * 0.5) + text_frame.shape[1]] = text_frame
        
        # Get class output
        out_tensor = interpreter.get_tensor(OUTPUT_IDX['class'])
        out_tensor = np.expand_dims(out_tensor, axis=0)
        out_tensor = np.expand_dims(out_tensor, axis=0)
        output_class = parseOutput(out_tensor, GRID_WIDTH['class'], int(width * SCREEN_WIDTH['class']), int(height * (1 - SPACING * 2)), name='class', border_width=1, border_color=0, frame_border_color=BORDER_COLOR, show_cell_border=True, normalize=False, apply_relu=False, apply_sigmoid=True, min_value=32)
        output_class_posX = output_pooling_posX + output_pooling.shape[1] + int(width * SPACING)
        output_class_posY = int(height * SPACING)
        frame[output_class_posY:output_class_posY + output_class.shape[0], output_class_posX:output_class_posX + output_class.shape[1]] = output_class

        # Add text on class
        text_frame = np.zeros((int(width * SPACING * 0.5), int(height * 0.5), IMAGE_CHANNELS), np.uint8)
        class_text = "Class output, {} species".format(out_tensor.shape[-1])
        cv2.putText(text_frame, class_text, (0, int(width * SPACING * 0.25)), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, TEXT_COLOR, 1, cv2.LINE_AA)
        text_frame = np.rot90(text_frame, 3)
        frame[output_class_posY:output_class_posY + text_frame.shape[0], output_class_posX - int(width * SPACING * 0.5):output_class_posX - int(width * SPACING * 0.5) + text_frame.shape[1]] = text_frame

        # Get N highest scoring classes with labels
        scores = interpreter.get_tensor(OUTPUT_IDX['class'])
        scores = 1 / (1 + np.exp(-scores))
        scores = scores[0]
        topN = np.argsort(scores)[::-1][:NUMBER_OF_RESULTS]
        topN_scores = scores[topN]
        topN_labels = [LABELS[i] for i in topN]
        
        # Show results
        for i in range(NUMBER_OF_RESULTS):
            bar_v_spacing = int(height * 0.025)
            bar_width = int(width * SCREEN_WIDTH['bar_width'])
            bar_height = int(height * (1 - SPACING * 1) / NUMBER_OF_RESULTS) - bar_v_spacing
            bar_posX = output_class_posX + output_class.shape[1] + int(width * SPACING)
            bar_posY = int(height * SPACING) + i * (bar_height + bar_v_spacing)

            # Draw bar background
            bc = (32, 32, 32) if topN_scores[i] > 0.3 else (32, 32, 32)
            cv2.rectangle(frame, (bar_posX, bar_posY), (bar_posX + bar_width, bar_posY + bar_height), bc, -1)   

            # Draw bar foreground based on score
            cv2.rectangle(frame, (bar_posX, bar_posY), (bar_posX + int(bar_width * topN_scores[i]), bar_posY + bar_height), TEXT_COLOR, -1)   

            # Draw label
            tc = max(32, min(255, topN_scores[i] * 2 * 255))
            tc = (tc, tc, tc)
            cv2.putText(frame, topN_labels[i], (bar_posX + bar_width + int(width * SPACING * 0.5), bar_posY + int(bar_height * 0.75)), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, tc, 1, cv2.LINE_AA)   

        # Add text on results
        text_frame = np.zeros((int(width * SPACING * 0.5), int(height * 0.5), IMAGE_CHANNELS), np.uint8)
        results_text = "Top {} results".format(NUMBER_OF_RESULTS)
        cv2.putText(text_frame, results_text, (0, int(width * SPACING * 0.25)), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, TEXT_COLOR, 1, cv2.LINE_AA)
        text_frame = np.rot90(text_frame, 3)
        frame[output_class_posY:output_class_posY + text_frame.shape[0], bar_posX - int(width * SPACING * 0.5):bar_posX - int(width * SPACING * 0.5) + text_frame.shape[1]] = text_frame

        # Show image in window
        cv2.imshow('demo', frame)

        # Save frame to file
        #cv2.imwrite('output.png', frame)        

        # Wait 1ms for user input
        key = cv2.waitKey(1)

        # If key is 'c', change channels
        if key == ord('c'):
            if COLORMAP == None:
                IMAGE_CHANNELS = 3
                TEXT_COLOR = (0, 255, 255)
                COLORMAP = cv2.COLORMAP_VIRIDIS
            elif COLORMAP == cv2.COLORMAP_VIRIDIS:
                IMAGE_CHANNELS = 3
                TEXT_COLOR = (164, 255, 255)
                COLORMAP = cv2.COLORMAP_INFERNO
            elif COLORMAP == cv2.COLORMAP_INFERNO:
                IMAGE_CHANNELS = 3
                TEXT_COLOR = (255, 255, 255)
                COLORMAP = cv2.COLORMAP_BONE
            else:
                IMAGE_CHANNELS = 1
                TEXT_COLOR = (255, 255, 255)
                COLORMAP = None

        # If key is 'p', pause
        elif key == ord('p'):
            PAUSE = not PAUSE
            cv2.waitKey(-1)
            PAUSE = not PAUSE

        # if key is 's', save image
        if key == ord('s'):
            cv2.imwrite('output.png', frame)

        # if key is 'a' switch to next audio file
        if key == ord('a'):

            global AUDIO_INDEX, STREAM, STREAM_WORKER

            # Stop stream
            STREAM = False

            # Wait for stream to finish
            STREAM_WORKER.join()

            # Load next audio file
            AUDIO_INDEX += 1
            if AUDIO_INDEX >= len(AUDIO_DATA):
                AUDIO_INDEX = 0

            # Start stream
            STREAM = True
            STREAM_WORKER = Thread(target=play, args=())
            STREAM_WORKER.start()

        # If user press ESC, break loop
        elif key == 27:
            break    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="BirdNET-XRay Demo.")
    parser.add_argument('--resolution', type=str, default='fullscreen', help='Resolution of the window, e.g., "fullscreen" or "1024x768"')
    parser.add_argument('--scaling', type=float, default='1.5', help='Scaling factor for the width of the output elements. Default is 1.5, lower values might work better on smaller screens.')
    parser.add_argument('--fontsize', type=float, default='0.55', help='Font size for text elements. Default is 0.55.')

    args = parser.parse_args()

    # Set font size
    FONT_SIZE = args.fontsize

    # Set resolution
    if args.resolution == 'fullscreen':
        frame_width = -1
        frame_height = -1
    else:
        try:
            frame_width, frame_height = map(int, args.resolution.split('x'))
        except ValueError:
            print("Invalid resolution format. Use 'fullscreen' or 'widthxheight'.")

    # Load data, model and window
    load(frame_width, frame_height, args.scaling)

    # Start recording
    STREAM = True
    STREAM_WORKER = Thread(target=play, args=())
    STREAM_WORKER.start()

    # Start main loop
    try:
        main()
    except:
        tb.print_exc()

    # Destroy window
    cv2.destroyWindow('demo')

    # Stop recording
    STREAM = False


