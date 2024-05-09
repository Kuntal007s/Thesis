import cv2
import numpy as np
import pandas as pd

def empty(a):
    pass

cv2.namedWindow('parameters')
cv2.resizeWindow('parameters', 640, 240)
cv2.createTrackbar('threshold1', 'parameters', 233, 255, empty)
cv2.createTrackbar('threshold2', 'parameters', 75, 255, empty)

def get_contours(img, imgcontour):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    all_data = []  # List to store all contour data for Excel
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 200 :
            M = cv2.moments(cnt)
            cX = int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0
            cY = int(M["m01"] / M["m00"]) if M["m00"] != 0 else 0
            cv2.circle(imgcontour, (cX, cY), 2, (255, 255, 255), -1)
            cv2.drawContours(imgcontour, [cnt], -1, (0, 255, 255), 2)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            # Extract vertices and calculate vectors
            vertices = approx[:, 0, :]
            vectors = np.diff(vertices, axis=0, append=[vertices[0]])
            vector_lengths = np.linalg.norm(vectors, axis=1)

            if vertices.ndim == 2 and len(vertices) > 1:
                 #Compute the step length tensor using the outer product of displacement vectors
                tensor = sum(np.outer(d, d) for d in vectors)
                
                # Compute eigenvalues and eigenvectors
                eigenvalues, eigenvectors = np.linalg.eig(tensor)


            # Check if the perimeter is greater than zero
            if peri > 0:
        # Calculate extent of circularity
                circularity = (4 * np.pi * area) / (peri ** 2)
        
        # Print the circularity of the contour
               # print("Circularity:", circularity)

            # Additional properties
            x, y, w, h = cv2.boundingRect(approx)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area != 0 else 0
            aspect_ratio = w / h if h != 0 else 0
            mean_vector_length = np.mean(vector_lengths, axis=0)

            # Append data for each contour
            all_data.append({
                "Centroid X": cX,
                "Centroid Y": cY,
                "Area": area,
                "Circularity": circularity,
                "Perimeter": peri,
                "Width": w,
                "Height": h,
                "Solidity": solidity,
                "Aspect Ratio": aspect_ratio,
                "vertices": vertices,
                "vectors": vectors,
                "Tensor for contour": tensor,
                "Eigenvalues": eigenvalues,
                "Eigenvector_1": eigenvectors[:, 0],
                "Eigenvector_2": eigenvectors[:, 1],
                "Vector Lengths": vector_lengths,
                "Mean vector length": mean_vector_length.tolist()
            })

    # Create DataFrame from collected data
    df = pd.DataFrame(all_data)
    df.to_excel(r'D:\mtech\New folder\dataseta.xlsx', index=False)  # Save to Excel
    print(r"Data saved to Excel file D:\mtech\New folder\dataset.xlsx")

while True:
    img = cv2.imread(r'D:\mtech\iit\cellpose\sam30.jpg')
    imgcontour = img.copy()
    imgBlur = cv2.GaussianBlur(img, (7, 7), 1)
    imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)

    threshold1 = cv2.getTrackbarPos('threshold1', 'parameters')
    threshold2 = cv2.getTrackbarPos('threshold2', 'parameters')
    imgcanny = cv2.Canny(imgGray, threshold1, threshold2)

    get_contours(imgcanny, imgcontour)

    cv2.imshow('result', img)
    cv2.imshow('contour', imgcontour)
    cv2.imshow('canny', imgcanny)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
