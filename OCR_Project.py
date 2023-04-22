import re
import cv2
import pytesseract
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import easyocr 
import pandas as pd
import datetime
import os

from skimage.filters import threshold_local
from PIL import Image
from pytesseract import Output



os.environ['KMP_DUPLICATE_LIB_OK']='True'


#1. Resize image (finding receipt contour is more efficient on a small image)
def opencv_resize(image):
    resize_ratio = 500 / image.shape[0]
    width = int(image.shape[1] * resize_ratio)
    height = int(image.shape[0] * resize_ratio)
    dim = (width, height)
    resized_image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return resized_image

#2. GrayScale
def To_gray(image):
     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
     return gray

#3. GaussianBlur
def To_GaussianBlur(image):
     blur = cv2.GaussianBlur(image, (5, 5), 0)
     return blur 

#4. Detect white regions
def rectKernel(image):
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    dilated = cv2.dilate(image, rectKernel)
    return dilated

#5. Canny Edge Detection & Contours
def edge_detection(image):
    edged = cv2.Canny(image, 100, 200, apertureSize=3)
    return edged
def get_contours(image,img):
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    image_with_contours = cv2.drawContours(img.copy(), contours, -1, (0,255,0), 3)
    return image_with_contours
def get_10_contours(image,img):
    largest_contours = sorted(image, key = cv2.contourArea, reverse = True)[:10]
    image_with_largest_contours = cv2.drawContours(img.copy(), largest_contours, -1, (0,255,0), 3)
    return image_with_largest_contours
def get_receipt_contour(contours):    
    # loop over the contours
    for c in contours:
        approx = approximate_contour(c)
    # if our approximated contour has four points, we can assume it is receipt's rectangle
        if len(approx) == 4:
            return approx
def approximate_contour(contour):
    peri = cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, 0.032 * peri, True)
                
#6. Original receipt with wrapped perspective
def wrap_perspective(img, rect):
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, M, (maxWidth, maxHeight))

#7. Convert 4 points into lines / rect   
def contour_to_rect(contour):
    pts = contour.reshape(4, 2)
    rect = np.zeros((4, 2), dtype = "float32")
    #7.1 top-left point has the smallest sum
    #7.2bottom-right has the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    #7.3 compute the difference between the points:
    #7.4 the top-right will have the minumum difference 
    #7.5 the bottom-left will have the maximum difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect /(500 / img.shape[0])

# 8. Threshold image
def bw_scanner(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    T = threshold_local(gray, 21, offset = 5, method = "gaussian")
    return (gray > T).astype("uint8") * 255
        
# 9. Find Amount
def find_amounts(text):
    amounts = re.findall(r'\d+\.\d{2,3}\b', text)
    floats = [float(amount) for amount in amounts]
    unique = list(dict.fromkeys(floats))
    return unique


pytesseract.pytesseract.tesseract_cmd = 'C:\\Users\\LT62182\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe'
# Add CSS to change the background color of the page

upload_image=st.sidebar.file_uploader('Choose an image',type=["jpg","png","jpeg"])

tab1,tab2,tab3 = st.tabs(["OCR","OCR Process","Database"])
  
with tab2:
    st.title("Reciept OCR APP Process")
    st.text("1. Upload a Reciept image which contains English Text")
    st.text("2. Upload a Reciept image wich non complexity background colors")

    if upload_image is not None:
            # Read image
            img = cv2.imdecode(np.fromstring(upload_image.read(), np.uint8), 1)

            # Display original image
            st.image(img, caption="1. Original Image", use_column_width=True)

            #Resize
            Re= opencv_resize(img)
            st.image(Re, caption="2. Resize Image", use_column_width=True)

            # Convert to grayscale
            gray = To_gray(Re)
            st.image(gray, caption="3. Grayscale Image", use_column_width=True)

            # Apply Gaussian Blur filter
            blur=To_GaussianBlur(gray)
            st.image(blur, caption="4. Blurred Image by Gaussian Blur filter", use_column_width=True)

            # Detect white regions
            dilated= rectKernel(blur)
            st.image(dilated, caption="5. Morphological Image for Detect white regions by Dilated medthod", use_column_width=True)

            # edge_detection
            edged = edge_detection(dilated)
            st.image(edged, caption="6. Edge detection Image", use_column_width=True)
            
            # Detect all contours
            contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            image_with_contours = cv2.drawContours(Re.copy(), contours, -1, (0,255,0), 3)
            st.image(image_with_contours, caption="7. Image with all contours", use_column_width=True)

            # Detect max 10 contours
            largest_contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
            image_with_largest_contours = cv2.drawContours(Re.copy(), largest_contours, -1, (0,255,0), 3)
            st.image(image_with_largest_contours, caption="8. Image with 10 largest contours", use_column_width=True)
            
            # Detect squred
            get_receipt_contour(largest_contours)
            receipt_contour = get_receipt_contour(largest_contours)
            image_with_receipt_contour = cv2.drawContours(Re.copy(), [receipt_contour], -1, (0, 255, 0), 2)
            st.image(image_with_receipt_contour, caption="9. Reciept contours Image", use_column_width=True)

            # Cropping
            cropping = wrap_perspective(img.copy(), contour_to_rect(receipt_contour))
            st.image(cropping , caption="10. Cropping Image", use_column_width=True)

            #Equaliazation
            #equalized = hist_equalization(cropping)
            #t.image(equalized, caption="Equaliazation Image", use_column_width=True)

            #Scanner
            result = bw_scanner(cropping)
            st.image(result , caption="11. Scanned Image", use_column_width=True)

            #Box 
            d = pytesseract.image_to_data(result, output_type=Output.DICT)
            n_boxes = len(d['level'])
            boxes = cv2.cvtColor(result.copy(), cv2.COLOR_BGR2RGB)
            for i in range(n_boxes):
                (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])    
                boxes = cv2.rectangle(boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
            st.image(boxes , caption="12. Box Image", use_column_width=True)


            #if st.button("Extract Text"):
            st.write("---------------------Extracted Text From Reciept By Pytesseract----------------------------")
            output_text=pytesseract.image_to_string(result)
            value = output_text.replace(',', '')
            value1 = value.replace('THB', ' ')
                #value2 = value1.replace('TOTAL', ' ')
            st.write(value1)
            amounts = find_amounts(value1)

            st.write("---------------------Extracted Amount From Text By Regular expression----------------------------")
            st.text("Amount of this reciept is :")
            st.write(max(amounts))
                #st.write(f"Amount of this receipt is: {max(amounts)}")
            st.write("------------------------------------------------------------------------------")

                #st.write("---------------------Extracted Text From Reciept By P----------------------------")
                #output_text1=pytesseract.image_to_string(cropping) 
                #amount = re.search(r'\d{1,3}(,\d{3})*\.\d{2}', output_text1).group() 
                #st.write(amount)

with tab1:
    st.title("Information")
    st.text("1. Browse receipt image")
    st.text("2. Input your Name ")
    st.text("3. Select your Department Name")
    st.text("4. Input Date in reciept")
    st.text("5. Submit Information")

    st.text_input("Name")

    option = st.selectbox(
        'Department Name',
        ('Select','D1', 'D2', 'D3'))
    
    d = st.date_input(
        "Date in reciept",
        datetime.date(2023, 4, 23))
     
    if upload_image is not None:
        st.image(img, caption="Your Receipt Image", use_column_width=True)
        st.write("---------------------Extracted Amount From Receipt----------------------------")
        st.text("Amount of this reciept is :")
        New_Amount = st.text_input("Amount of this reciept is :",max(amounts))
        st.write("------------------------------------------------------------------------------")

        #DB Mapping
        if st.button("Submit Information") : 
            st.write("connect DB")

with tab3:
    @st.cache_data
    def load_data(sheets_url):
        csv_url = sheets_url.replace("/edit#gid=", "/export?format=csv&gid=")
        return pd.read_csv(csv_url)

    df = load_data(st.secrets["public_gsheets_url"])
    # Print results.
    st.dataframe(df)



            