# Subject-level adjacency matrices are made availabe at https://doi.org/10.6084/m9.fgshare.6983018
# Code is available at https://github.com/faskowit/Faskowitz2018wsbmLifeSpan


import nibabel as nib

# Define the path to the NIfTI file
file_path = r'C:\Users\barbo\Desktop\thesis repo clone\thesis\data\BNA_SC_4D.nii\BNA_SC_4D.nii'

# Load the NIfTI file
img = nib.load(file_path)

# Access the data
data = img.get_fdata()

# Print the shape of the data to understand its dimensions
print(data.shape)

# Optionally, you can print a small part of the data to inspect it
print(data.shape)  # Output: (91, 109, 91, 246)

# Access data for the first subject
subject_index = 0
subject_data = data[..., subject_index]

# Print the shape of the subject data to confirm
print(subject_data.shape)  # Output: (91, 109, 91)

# Optionally, visualize a slice of the subject data
import matplotlib.pyplot as plt

slice_index = 45  # Adjust the index as needed
plt.imshow(subject_data[:, :, slice_index], cmap='gray')
plt.title(f'Slice {slice_index} for Subject {subject_index}')
plt.colorbar()
plt.show()