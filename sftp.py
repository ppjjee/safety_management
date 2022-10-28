import pysftp
import os.path
import os

class SFTP():

    def __init__(self, hostname, username, password):
        # self.dirRemoteMusicData = "/nas3/epark/workspace/retreival/music_data/mp3"
        # self.dirRemoteSurveyResult = "/nas3/epark/workspace/retreival/music_data/survey"
        self.dirModel = "total_EfficientNetB4_revision.h5"
        
        cnopts = pysftp.CnOpts()
        cnopts.hostkeys = None
        self.sftp = pysftp.Connection(host = hostname, username = username, password = password, cnopts = cnopts)
        print("connection successfully established...")
        # if not os.path.exists(self.dirModel):
        #     os.makedirs(self.dirModel)
        
    def upload(self, localFilePath, remoteFilePath):
        if not os.path.exists(localFilePath):
            print("not exists. " + localFilePath)
            return
        if self.sftp.exists(remoteFilePath):
            print("already exists. " + remoteFilePath)
            return
        self.sftp.put(localFilePath, remoteFilePath)

    def download(self, remoteFilePath, localFilePath):
        if os.path.exists(localFilePath):
            return
        self.sftp.get(remoteFilePath, localFilePath)



    
