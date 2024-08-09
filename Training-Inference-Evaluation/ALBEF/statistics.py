import cv2
import numpy as np
import os


def calculate_mean_std(image_folder):
   # Initialize the sum and squared sum for each channel
   channels_sum = np.zeros(3)
   channels_squared_sum = np.zeros(3)
   num_pixels = 0


   for filename in os.listdir(image_folder):
       if filename.endswith('.png'):
           img_path = os.path.join(image_folder, filename)
           img = cv2.imread(img_path)


           if img is None:
               continue


           # Convert the image to float32 and normalize pixel values to [0, 1]
           img = img.astype(np.float32) / 255.0


           # Sum the pixel values and the squared pixel values for each channel
           channels_sum += np.sum(img, axis=(0, 1))
           channels_squared_sum += np.sum(img ** 2, axis=(0, 1))
           num_pixels += img.shape[0] * img.shape[1]


   # Calculate the mean and standard deviation
   mean = channels_sum / num_pixels
   std = np.sqrt(channels_squared_sum / num_pixels - mean ** 2)


   return mean, std


if __name__ == "__main__":
   image_folder = '/home/avnish/Downloads/archive/images/images_normalized/'
   mean, std = calculate_mean_std(image_folder)
   print(f'Mean: {mean}')
   print(f'Std Deviation: {std}')