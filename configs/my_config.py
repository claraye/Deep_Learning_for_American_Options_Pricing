import os
import sys
sys.path.append('util')
from file_system import WindowsFS, GcsFS
try:
    from datalab.context import Context
    import google.datalab.storage as storage
    from io import BytesIO
except ImportError:
    print('Not Running in Google Datalab')

class Config:
    FS = WindowsFS()
    #FS = GcsFS()
    
    if isinstance(FS, WindowsFS):
        data_dir = r"C:\_Documents\COURSES\Fall 2019\IEOR4742 Deep Learning\Project\Final\Data"
        datasets_dir = os.path.join(data_dir,'datasets')
        experiments_dir = r"C:\_Documents\COURSES\Fall 2019\IEOR4742 Deep Learning\Project\Final\Experiments"
    elif isinstance(FS, GcsFS):
        mybucket_name = Context.default().project_id
        FS.set_bucket_name(mybucket_name)
        
        #data_dir = 'gs://' + mybucket_name + '/Data'
        data_dir = 'Data'
        datasets_dir = os.path.join(data_dir,'datasets')
        experiments_dir = 'Experiments'