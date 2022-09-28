from pathlib import Path
import streamlit as st
import subprocess
from detect import detect 
import os
import argparse
from PIL import Image


def get_subdirs(b='.'):
    '''
        Returns all sub-directories in a specific Path
    '''
    result = []
    for d in os.listdir(b):
        bd = os.path.join(b, d)
        if os.path.isdir(bd):
            result.append(bd)
    return result


def get_detection_folder():
    '''
        Returns the latest folder in a runs\detect
    '''
    return max(get_subdirs(os.path.join('runs', 'detect')), key=os.path.getmtime)

# For Webcam



if __name__ == '__main__':

    st.title('YOLOv5 Streamlit App')

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,
                        default='mymask.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str,
                        default='data/images', help='source')
    parser.add_argument('--img-size', type=int, default=640,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float,
                        default=0.35, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true',
                        help='display results')
    parser.add_argument('--save-txt', action='store_true',
                        help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true',
                        help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true',
                        help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument('--update', action='store_true',
                        help='update all models')
    parser.add_argument('--project', default='runs/detect',
                        help='save results to project/name')
    parser.add_argument('--name', default='exp',
                        help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    opt = parser.parse_args()


    source = ("Image", "Video")
    source_index = st.sidebar.selectbox("Select Your Files", range(
        len(source)), format_func=lambda x: source[x])

    if source_index == 0:
        uploaded_file = st.sidebar.file_uploader(
            "Select Image", type=['png', 'jpeg', 'jpg'])
        if uploaded_file is not None:
            is_valid = True
            with st.spinner(text='Loading Image...'):
                st.sidebar.image(uploaded_file)
                picture = Image.open(uploaded_file)
                picture = picture.save(f'{uploaded_file.name}')
                opt.source = f'{uploaded_file.name}'
        else:
            is_valid = False
    else:
        uploaded_file = st.sidebar.file_uploader("Select Video", type=['mp4'])
        if uploaded_file is not None:
            is_valid = True
            with st.spinner(text='Loading Video...'):
                uploaded_file.name
                st.sidebar.video(uploaded_file)
                with open(os.path.join(uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())
                opt.source = f'{uploaded_file.name}'
        else:
            is_valid = False
        
        
            
        
    
    
    if is_valid:
        
        if st.button('Detect'):
            
            detect(opt)
    
            if source_index == 0:
                with st.spinner(text='Preparing Images'):
                    for img in os.listdir(get_detection_folder()):
                        st.image(str(Path(f'{get_detection_folder()}') / img))

                    st.balloons()
            
            else :
                with st.spinner(text='Preparing Video'):
                    for vid in os.listdir(get_detection_folder()):
                        vid_path = str(Path(f'{get_detection_folder()}') / vid)
                        
                        cmd1 = f"ffmpeg -i {vid_path} -vcodec libx264 -acodec aac newencoded.mp4"
                        subprocess.Popen(cmd1, shell=True)
            
                        st.video(str("newencoded.mp4"))
                    st.balloons()
                   
        
   

                  

                        
            
           
        
                    
                



                    