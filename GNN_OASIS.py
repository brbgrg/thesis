import pyxnat
import requests
import os

# Define XNAT server URL and credentials
XNAT_URL = 'https://central.xnat.org'
USERNAME = 'your_username'
PASSWORD = 'your_password'

# Connect to the XNAT server
interface = pyxnat.Interface(server=XNAT_URL, user=USERNAME, password=PASSWORD)

# Define the project and dataset
project_id = 'OASIS1'
dataset_id = 'OAS1_0001_MR1'

# Query the dataset
project = interface.select.project(project_id)
subject = project.subject(dataset_id)

# Define the download directory
download_dir = 'oasis1_dataset'
os.makedirs(download_dir, exist_ok=True)

# Download the dataset files
for scan in subject.experiment('MR1').scans():
	scan_id = scan.id()
	resource = scan.resource('DICOM')
	files = resource.files()
	for f in files:
		file_url = f.uri()
		file_name = os.path.join(download_dir, f.name())
		response = requests.get(XNAT_URL + file_url, auth=(USERNAME, PASSWORD))
		with open(file_name, 'wb') as file:
			file.write(response.content)
		print(f'Downloaded {file_name}')

print('Download complete.')