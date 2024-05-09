import cv2
import numpy as np
import matplotlib.pyplot as plt

# def compute_step_length_tensor(vertices):
#     # Calculate displacement vectors
#     displacement_vectors = np.diff(vertices, axis=0, append=[vertices[0]])
#     # Compute the step length tensor using the outer product of displacement vectors
#     tensor = sum(np.outer(d, d) for d in displacement_vectors)
#     return tensor


def empty(a):
    pass

cv2.namedWindow('parameters')
cv2.resizeWindow('parameters', 640, 240)
cv2.createTrackbar('threshold1', 'parameters', 233,255, empty)
cv2.createTrackbar('threshold2', 'parameters', 75,255, empty)

# Initialize lists to store tensors and eigenvalues from all contours
all_tensors = []
approx_points_count = []


def getcontours (img, imgcontour):
    global all_tensors  

    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # cv2.drawContours(imgcontour, contours, -1, (0, 0, 255), 5)
    # print('number of contours found = {}'.format(len(contours)))

    

    for cnt in contours:
           
        
        
                
        area = cv2.contourArea(cnt)
        if area>200:
        # if area>200:
            M = cv2.moments(cnt)
            cX = int(M["m10"]/M["m00"])
            cY = int(M["m01"]/M["m00"])

             

            cv2.circle(imgcontour, (cX, cY), 2, (255, 255, 255), -1)
            print("centroid coordinates:", (cX, cY))
            # cv2.drawContours(imgcontour, contours, -1, (0, 0, 0), 3)
            cv2.drawContours(imgcontour, contours, -1, (0, 255, 255), 2)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
            approx_points_count.append(len(approx))  # Store the number of points in the approx
            print(f"Points: {len(approx)}")
            print(f"Contour Area: {area}")

    
    
            # Extract vertices and calculate vectors
            vertices = approx[:, 0, :]  # Reshape array
            vectors = np.diff(vertices, axis=0, append=[vertices[0]])      
            
           

            # tensor = compute_step_length_tensor(vertices)  # Compute the tensor
            # print(f"Tensor for Contour:\n{tensor}\n")

            print("Vertices:", vertices)
            print("Vectors:", vectors)
            
            
            
            if vertices.ndim == 2 and len(vertices) > 1:
                 #Compute the step length tensor using the outer product of displacement vectors
                tensor = sum(np.outer(d, d) for d in vectors)
                
                # Compute eigenvalues and eigenvectors
                eigenvalues, eigenvectors = np.linalg.eig(tensor)

                all_tensors.append(tensor)
                
                print(f"Tensor for Contour:\n{tensor}\n")

                print("Eigenvalues:", eigenvalues)
                print("Eigenvectors:\n", eigenvectors)
            

            # Calculate the lengths of each vector
            vector_lengths = np.linalg.norm(vectors, axis=1)
            print("Vector Lengths:", vector_lengths)

            mean_vector_length = np.mean(vector_lengths, axis=0)
            print("mean vector length", mean_vector_length)
            

            # Check if the perimeter is greater than zero
            if peri > 0:
        # Calculate extent of circularity
                circularity = (4 * np.pi * area) / (peri ** 2)
        
        # Print the circularity of the contour
                print("Circularity:", circularity)
            # print("number of contours:", len(contours))

            # for point in cnt:
            #     x, y = point[0]
            #     print('coordinates:', (x, y))
            
         

                      

            x, y, w, h =cv2.boundingRect(approx)

            # Perimeter
            perimeter = cv2.arcLength(cnt, True)

            # Feret's diameter and angle
            (min_feret_rect, (min_feret_width, min_feret_height), feret_angle) = cv2.minAreaRect(cnt)
            feret_diameter = max(min_feret_width, min_feret_height)

            # Roundness (also known as compactness or circularity)
            # area = cv2.contourArea(cnt)
            # roundness = 4 * np.pi * area / (perimeter ** 2) if perimeter != 0 else 0

            # Solidity
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area != 0 else 0

            # Aspect Ratio
            aspect_ratio = w / h if h != 0 else 0

            # Print the results
            # print(f"Contour Properties:")
            # print(f"Centroid: ({cX}, {cY})")
            print(f"Width: {w}, Height: {h}")
            print(f"Perimeter: {perimeter}")
            print(f"Feret's Diameter: {feret_diameter}, Feret's Angle: {feret_angle}")
            # print(f"Roundness: {roundness}")
            print(f"Solidity: {solidity}")
            print(f"Aspect Ratio: {aspect_ratio}")
            print("--------")

            # cv2.rectangle(imgcontour, (x, y), (x+w, y+h), (0, 255, 0), 5)
            
            # cv2.putText(imgcontour, 'points: ' + str(len(approx)), (x+w+5, y+5), cv2.FONT_HERSHEY_COMPLEX, .7, (0,255,0),2)
            # cv2.putText(imgcontour, 'area: ' + str(int(area)),(x+w+5, y+15), cv2.FONT_HERSHEY_COMPLEX, .7, (0,255,0),2)


    

            # mean_tensor = np.mean(tensor, axis=0)
            # print("mean tensor", mean_tensor)

            # mean_eigen = np.mean(eigenvalues, axis=0)
            # print("mean eigen", mean_eigen)
    if approx_points_count:
        average_points = np.mean(approx_points_count)
        print(f"Average number of points: {average_points:.2f}")
        approx_points_count.clear()  # Clear the list after processing for fresh calculations next time        


while True:
    img = cv2.imread(r'D:\mtech\iit\cellpose\sam30.jpg')
    # img = cv2.imread(r'D:\mtech\iit\CELL IMAGES\new2.jpg')
    imgcontour = img.copy()
    imgBlur = cv2.GaussianBlur(img, (7,7), 1)
    imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)

    threshold1 = cv2.getTrackbarPos('threshold1', 'parameters')
    threshold2 = cv2.getTrackbarPos('threshold2', 'parameters')

    imgcanny = cv2.Canny(imgGray, threshold1, threshold2)
    # kernel = np.ones((5,5))
    # imgDil = cv2.dilate(imgcanny, kernel, iterations=1)
        
    getcontours(imgcanny, imgcontour)


    # plt.figure()
    # plt.imshow(imgcontour)
    
    cv2.imshow('result', img)
    cv2.imshow('contour',imgcontour)
    cv2.imshow('canny',imgcanny)
    # cv2.imshow('dilate',imgDil)


    #Saving the images 
    # result = cv2.imwrite(r'D:\mtech\New folder\sam30a.png', imgcontour)
    # result = cv2.imwrite(r'D:\mtech\New folder\sam30b.png', imgcanny)

    if all_tensors:
        mean_tensor = np.mean(all_tensors, axis=0)
        mean_eigenvalues, mean_eigenvectors = np.linalg.eig(mean_tensor) 
        print("Mean Tensor:\n", mean_tensor)
        print("Eigenvalues of the Mean Tensor:", mean_eigenvalues)
        print("Eigenvectors of the Mean Tensor:", mean_eigenvectors)
        
        all_tensors = []  # Clear the list for the next image
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break