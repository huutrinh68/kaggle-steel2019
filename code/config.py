class Config:
    def __init__(self):
        # train hyperparameters setting
        self.n_classes = 4
        self.img_size = 224
        self.batch_size = 2
        self.lr = 1e-4
        self.test_size = 0.15
        self.random_state = 2019
        self.n_workers = 16
        self.n_epochs = 400
        self.best_avg_loss = 100
        self.weight_decay = 1e-5
        self.n_step = 5
        self.gamma = 0.75
        self.n_splits = 5
        self.patience = 10

        # test hyperparameters setting
        self.n_tta = 1

        # output setting
        self.model_name = "efficientb3"
        self.output_path = "../output/"
        self.best_model = "../output/best_model_efficientb3/"
        self.logs = "../output/logs/"
        self.train_loss = "data/train_loss"
        self.train_f1 = "data/train_f1"
        self.eval_loss = "data/eval_loss"
        self.eval_f1 = "data/eval_f1"

        # input data setting
        self.train_img = "../input/severstal-steel-defect-detection/train_images/"
        self.test_img = "../input/severstal-steel-defect-detection/test_images/"
        self.train_csv = "../input/severstal-steel-defect-detection/train.csv"
        self.sample_submission_csv = "../input/severstal-steel-defect-detection/sample_submission.csv"

config = Config()