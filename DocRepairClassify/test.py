import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import para
img=mpimg.imread(para.data_result_path+'/RoI_results/RoI.png')
plt.figure()
plt.imshow(img) 
plt.show()