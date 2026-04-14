from PIL import Image
import matplotlib.pyplot as plt

img_path = '/data2/data/cropped_Tesla/integrated/Volvo_240_Sedan_1993/train_007857.jpg'
img = Image.open(img_path)

plt.imshow(img)
plt.axis('off')
plt.show()