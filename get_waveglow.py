import sys
import warnings

from git import Repo
from google_drive_downloader import GoogleDriveDownloader as gdd

Repo.clone_from("https://github.com/NVIDIA/waveglow.git", "./waveglow/")
gdd.download_file_from_google_drive(
    file_id='1rpK8CzAAirq9sWZhe9nlfvxMF1dRgFbF',
    dest_path='./waveglow_256channels_universal_v5.pt'
)


sys.path.append('waveglow/')
warnings.filterwarnings('ignore')
