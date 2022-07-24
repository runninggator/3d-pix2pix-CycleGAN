from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        self.parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        self.parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--how_many', type=int, default=50, help='how many test images to run')
        self.parser.add_argument("--image", type=str, default='./Data_folder/test/images/0.nii')
        self.parser.add_argument("--result", type=str, default='./Data_folder/test/images/result_0.nii', help='path to the .nii result to save')
        self.parser.add_argument("--stride_inplane", type=int, nargs=1, default=32, help="Stride size in 2D plane")
        self.parser.add_argument("--stride_layer", type=int, nargs=1, default=32, help="Stride size in z direction")

        self.isTrain = False
