import shutil
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image, ImageTk
import glob
import os

#to test in python script

window = Tk()
window.title("GUI")
window.geometry("480x450")
window.configure(background='white')
window.resizable(0, 0)

image = Image.open("giphy.gif")
frames = []
try:
    while True:
        frames.append(ImageTk.PhotoImage(image))
        image.seek(len(frames))
except EOFError:
    pass

# Create a label with the image as the background
background_label = Label(window)
background_label.pack()

# Function to animate the GIF frames
def animate(index):
    frame = frames[index]
    background_label.configure(image=frame)
    window.after(100, animate, (index + 1) % len(frames))

# Start the animation
animate(0)

def file_upload():
    answer = filedialog.askopenfile(parent=window,
                                    initialdir=os.getcwd(),
                                    title="Please select image/Video:")
    #Label(window, text=answer.name, font=('Courier', 10), bg="white").place(x=50, y=130)
    return answer.name


def object_detection():
    model=YOLO("best.pt")
    img = file_upload()
    model.predict(show=True, conf=0.5, source=img, save_txt=True )
    
    latest_folder=glob.glob('runs\detect\*')
    f=max(latest_folder,key=os.path.getctime)
    myfile=glob.glob(f+'\labels\*')

    if (myfile!=[]):
        # Read the image
        img = cv2.imread(img)

        # Convert to HSV color space
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Define lower and upper bounds of water color in HSV
        lower_blue = np.array([80, 50, 50])
        upper_blue = np.array([130, 255, 255])

        # Threshold the image based on water color
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # Apply the mask to the original image
        masked_img = cv2.bitwise_and(img, img, mask=mask)

        cv2.imshow('masked img = ',masked_img)
        
    else:
        top = Tk()           
        messagebox.showinfo("Message","Oops! No waterbody present!")  
        top.mainloop()
       
    cv2.destroyAllWindows(1000)



def help_detection():
    Label(window, text='Upload video or image for object detection', bg="white", font=('Courier', 11)).place(x=200,y=200)


Label(window, text='Waterbody Detection and Masking', bg="violet", font=('Algerian', 15)).place(x=10, y=30)
#Label(window, text='Select an image for detection and masking of waterbody:-', font=('Courier', 10), bg="violet").place(x=0, y=400)

Button(window, text="Select Image", bg="violet",command=object_detection, font=('Algerian', 15)).place(x=290, y=380)

window.mainloop()
