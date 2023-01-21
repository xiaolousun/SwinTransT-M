class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/home/xlsun/xlsun/code/SLAT'   # Base directory for saving network checkpoints.
        self.tensorboard_dir = self.workspace_dir + '/tensorboard/'    # Directory for tensorboard files.
        self.lasot_dir = '/data01/xlsun/dataset/LaSOT/LaSOTBenchmark'
        self.got10k_dir = '/data/full_data/train'
        self.trackingnet_dir = '/home/cx/cx1/TrackingNet'
        self.coco_dir = '/data01/xlsun/dataset/coco2017'
        self.lvis_dir = ''
        self.sbd_dir = ''



        self.imagenet_dir = '/home/cx/cx3/ILSVRC2015'
        self.imagenetdet_dir = '/home/cx/cx3/Imagenet_DET/ILSVRC'
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
        self.youtube_vos_dir = '/home/cx/cx3/Youtube-VOS'
        self.saliency_dir = '/home/cx/cx3/saliency/MERGED'

