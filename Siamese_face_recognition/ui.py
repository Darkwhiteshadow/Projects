import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import cv2
import os
from PIL import Image
from mymodel import verify



def create_folder():
    global folder 

    output_folder = st.sidebar.text_input("Enter the name for the output folder", "ID")
    
    folder = os.path.join("./verification2", output_folder)

    webrtc_streamer(key="example", video_processor_factory=VideoRecorder)
    

class VideoRecorder(VideoTransformerBase):

    def __init__(self):
        self.frame_count = 0
        self.output_folder = folder

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
    
    
    def recv(self, frame):

        # Display frame
        st.image(frame.to_image(), caption=f'Frame {self.frame_count}', use_column_width=True)
        count=0
        # Save frame to the output folder
        img = frame.to_ndarray(format="bgr24")
        if(self.frame_count%30==0 and count<15):
            cv2.imwrite(f"{self.output_folder}/frame_{self.frame_count}.jpg",img)       
            count=count+1
        self.frame_count += 1
        
        return frame


# Streamlit app
def main():
    create_folder()
    st.sidebar.title("Image Classification App")
    

    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # Classify uploaded image
    if uploaded_file is not None:   
        with open(os.path.join("./input/", "image.png"), "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File '{uploaded_file.name}' uploaded successfully!")

        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Make predictions when button is clicked
        if st.sidebar.button('Classify'):
            # byte_img = tf.io.read_file()
            verified = verify("./input/image.png",folder, 0.5, 0.5)
            # st.image(verified)
            st.write(verified)

            


if __name__ == '__main__':
    
    main()
