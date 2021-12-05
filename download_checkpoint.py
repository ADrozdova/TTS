from google_drive_downloader import GoogleDriveDownloader as gdd
import os

GOOGLE_DRIVE_CHECKPOINT_ID = "1F8t5nerzg034BeilEYsQ5LCdg1vtBDWr"
GOOGLE_DRIVE_CONFIG_ID = "1Wmegs_dcdv3tl1d_HBAaqAxAa4hbUmJh"

DIR = "."


if not os.path.exists(DIR):
    os.mkdir(DIR)

gdd.download_file_from_google_drive(file_id=GOOGLE_DRIVE_CHECKPOINT_ID,
                                    dest_path=os.path.join(DIR, "checkpoint.pth"),
                                    unzip=False)
gdd.download_file_from_google_drive(file_id=GOOGLE_DRIVE_CONFIG_ID,
                                    dest_path=os.path.join(DIR, "config.json"),
                                    unzip=False)
