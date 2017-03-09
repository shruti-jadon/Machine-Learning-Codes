
import util
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA,FastICA,TruncatedSVD
import Plot_Denoise

path="C:/Users/Shruti Jadon/Documents"

#Set patch size -- change this to set the size of the patches that 
#the de-noising methods will operate over. The input image size is 
#900 rows x 1200 columns. The patch size is also listed as patch 
#height x patch width. The patch height and width must divide into
#the image width and weight evenly.
patch_size = (10,10)    # 90 X 120

#Load the data. data_noisy and data_clean are arrays with one
#row per image patch. The rows are formed by creating a single
#long vectors of size patch_size[0]*patch_size[1]*3 from the 
#color image patach of size patch_size[0]xpatch_size[1]. img_noisy
#and img_clean are the full clean and noisy imges.
data_noisy, data_clean, img_noisy, img_clean = util.loadDataQ2(patch_size)

#You should learn the dimensionality rediction-based de-noising models 
#on the *noisy* data data_noisy, and then use them to de-noise the noisy data. 
#This process is called reconstruction. Your de-noised data should 
#have the same shape as data_noisy (and data_clean), and should be
#placed in the array data_denoised as seen below. This starter
#code can be thought of as applying the identiity function as the de-noising
#operator.
errors={}
for values in range(1,64):
    pca = PCA(n_components=values) # declare PCA model
    pca.fit(data_noisy)             # fit model
    reduced_vector=pca.transform(data_noisy) # transform pca matrix
    data_denoised =pca.inverse_transform(reduced_vector) # inverse transform matrix
    #This function takes your de-denoised data and re-assembles it into a complete color image
    img_denoised  = util.patchToImage(util.vectorsToPatches(data_denoised,patch_size),(900,1200))
    errors[values]=round(util.eval_recon(img_clean,img_denoised),2)

#plotting of errors with different hyperparameter values
# Plot a bar chart
Plot_Denoise.plot(errors,"PCA",21,path)

print "Error of Noisy to Clean: %.4f"%util.eval_recon(img_clean,img_noisy)
print "Error of De-Noised to Clean: %.4f"%util.eval_recon(img_clean,img_denoised)

# Experimentation with Truncated SVD instead of PCA
errors={}
for values in range(1,64):
    svd = TruncatedSVD(n_components=values, n_iter=7, random_state=42)# declare model
    svd.fit(data_noisy)                                                # fit model
    reduced_vector=svd.transform(data_noisy)
    data_denoised =svd.inverse_transform(reduced_vector)   # inverse tranform to original vector
    img_denoised  = util.patchToImage(util.vectorsToPatches(data_denoised,patch_size),(900,1200))
    errors[values]=round(util.eval_recon(img_clean,img_denoised),2)   # store values

Plot_Denoise.plot(errors,"SVD",22,path)


#values with optimal number of clusters
B = np.random.rand(100,300)
pca = PCA(n_components=9) # declare PCA model
pca.fit(data_noisy)             # fit model
reduced_vector=pca.transform(data_noisy) # transform pca matrix
data_denoised =pca.inverse_transform(reduced_vector) # inverse transform matrix
#This function takes your de-denoised data and re-assembles it into a complete color image
img_denoised  = util.patchToImage(util.vectorsToPatches(data_denoised,patch_size),(900,1200))

#These functions compute the MAE between the clean image,the noisy image and the de-noised image
print "Error of Noisy to Clean: %.4f"%util.eval_recon(img_clean,img_noisy)
print "Error of De-Noised to Clean: %.4f"%util.eval_recon(img_clean,img_denoised)

#Plot the clean and noisy images
plt.figure(0,figsize=(7,3))
util.plot_pair(img_clean,img_noisy,"Clean","Noisy")

#Plot the clean and de-noised images
plt.figure(1,figsize=(7,3))
util.plot_pair(img_clean,img_denoised,"Clean","De-Noised")

#Plot the image basis
plt.figure(2,figsize=(4,3))
util.plot_img_array(B,patch_size)
plt.suptitle("Image Basis")
plt.show()

