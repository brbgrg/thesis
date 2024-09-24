import boto3
import os
import json

# Initialize a session using Amazon S3
s3 = boto3.client('s3')

# Define the HCP S3 bucket name
bucket_name = 'hcp-openaccess'

# List objects in the bucket
response = s3.list_objects_v2(Bucket=bucket_name, Prefix='HCP_1200/')

# Print the names of the first 10 objects
for obj in response.get('Contents', [])[100:200]:
	print(obj['Key'])


# List of JSON metadata file keys to download
json_file_keys = [
    'HCP_1200/100206/.xdlm/100206_3T_Structural_preproc.json',
    'HCP_1200/100206/.xdlm/100206_3T_Structural_preproc_extended.json',
    'HCP_1200/100206/.xdlm/100206_3T_rfMRI_REST1_preproc.json',
    'HCP_1200/100206/.xdlm/100206_3T_rfMRI_REST1_fixextended.json'
]

# Download each JSON metadata file
for json_file_key in json_file_keys:
    # Extract the base name of the file from the key
    json_file_name = os.path.basename(json_file_key)
    
    # Download the JSON metadata file
    s3.download_file(bucket_name, json_file_key, json_file_name)
    print(f'File {json_file_name} downloaded successfully')

    # Read the content of the JSON file
    with open(json_file_name, 'r') as json_file:
        json_content = json.load(json_file)
        
        # Extract and download files specified by the URIs in the JSON content
        for item in json_content:
            file_uri = item['URI']
            file_name = os.path.basename(file_uri)
            
            # Download the file specified by the URI
            s3.download_file(bucket_name, file_uri, file_name)
            print(f'File {file_name} downloaded successfully')



# preprocessing (Zhao2022a)
#fMRI:
# HCP minimal preprocessing pipeline [8]. 
# The artefacts of the BOLD signal were further removed using ICA-FIX. 
# The cortical surface was parcellated into N=360 major ROIs using MMP1.0 parcellation


