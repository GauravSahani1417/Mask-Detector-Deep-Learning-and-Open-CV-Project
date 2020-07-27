import streamlit as st 
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np 

@st.cache
def load_image(img):
    im = Image.open(img)
    return im

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
car_cascade = cv2.CascadeClassifier('haarcascade_car.xml')
ped_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')

model = load_model('model_MaskClassifier.h5')

labels_dict={0:'MASK WORE',1:'NO MASK'}
color_dict={0:(0,255,0),1:(255,0,0)}

def detect_faces(our_image):
    new_img = np.array(our_image.convert('RGB'))
    img = cv2.cvtColor(new_img,1)
    gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
                 face = cv2.resize(img, (94, 94))
                 face = img_to_array(face)
                 face = np.expand_dims(face, axis=0)
                 result=model.predict(face)
                 if result==1:
                     cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[0],2)
                     cv2.rectangle(img,(x,y-40),(x+w,y),color_dict[0],-1)
                     cv2.putText(img, labels_dict[0], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
                 else:
                     cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[1],2)
                     cv2.rectangle(img,(x,y-40),(x+w,y),color_dict[1],-1)
                     cv2.putText(img, labels_dict[1], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
                
                 
    return img,faces 

def detect_eyes(our_image):
    new_img = np.array(our_image.convert('RGB'))
    img = cv2.cvtColor(new_img,1)
    gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
    for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(img,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    return img


image = Image.open('mask.png')
st.image(image, use_column_width=True,format='PNG')

html_temp = """
    <div style="background-color:#010200;padding:6px">
    <h2 style="color:white;text-align:center;">Mask Detection Application</h2>
    </div>
    """    
st.markdown(html_temp,unsafe_allow_html=True)

def main():
    """Face Detection App"""
    activities = ["Mask Detection","About"]
    
    st.subheader("Select Activity :")
    choice = st.selectbox("", activities)

    if choice == 'Mask Detection':
        st.subheader("Mask Detection :")

        image_file = st.file_uploader("Upload Image :",type=['jpg','png','jpeg'])

        if image_file is not None:
            our_image = Image.open(image_file)
            st.text("Original Image")
            # st.write(type(our_image))
            st.image(our_image,width=600)
            
            
        task = ["Face","Eyes"]
        feature_choice = st.selectbox("Select Features :",task)
        if st.button("Run"):

            if feature_choice == 'Face':
                result_img,result_faces = detect_faces(our_image)
                st.image(result_img,width=600)

                st.success("Found {} faces".format(len(result_faces)))
                
            elif feature_choice == 'Eyes':
                result_img = detect_eyes(our_image)
                st.image(result_img,width=600)
                
                
            html_temp = """
                <div style="background-color:#010200;padding:4px">
                <h4 style="color:white;text-align:center;">Designed and Developed by: Gaurav Rajesh Sahani</h4>
                </div>
                """    
            st.markdown(html_temp,unsafe_allow_html=True)
            
    elif choice == 'About':
        st.subheader("About Mask Detection App")
        st.markdown("This Application is build training Deep Learning model using Convolutional Neural-Nets, augmented with OpenCV, have used Haar-cascades for eyes and frontal face detection.")
        st.markdown("Github Link for the code : [Code Link](https://github.com/GauravSahani1417)")
        
        st.subheader("Connect with me:")
        st.markdown("Designed and Developed by [Gaurav Rajesh Sahani](https://www.linkedin.com/in/gaurav-sahani-6177a7179/)")



if __name__ == '__main__':
        main()    



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
