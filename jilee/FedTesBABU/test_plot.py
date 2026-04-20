import matplotlib
# Use the Agg backend for remote servers without a GUI
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from PIL import Image

img_path = '/data2/data/cropped_Tesla/integrated/Volvo_240_Sedan_1993/train_007857.jpg'

try:
    img = Image.open(img_path)
    plt.imshow(img)
    plt.axis('off')
    # Save the file to your /data2 drive which has space
    plt.savefig('/home/jilee/jilee/FedTesBABU/utils/check_volvo.png') 
    print("Success! Image saved to /data2/data/check_volvo.png")
except Exception as e:
    print(f"Error: {e}")