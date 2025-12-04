from yacs.config import CfgNode as CN

class BaseConfig:
  def __init__(self):
    ###########################
    # Config definition
    ###########################
    cfg = CN(new_allowed=True)

    ###########################
    # Env
    ###########################
    cfg.ENV = CN(new_allowed=True)
    cfg.ENV.VERSION = 1
    cfg.ENV.SEED = -1
    # Directory to save the output files (like log.txt and model weights)
    cfg.ENV.OUTPUT_DIR = "./output"
    # Path to a directory where the files were saved previously
    cfg.ENV.RESUME = ""
    cfg.ENV.GPU = [0]
    cfg.ENV.USE_CUDA = True
    # Distributed training settings
    cfg.ENV.DISTRIBUTED = False
    cfg.ENV.LOCAL_RANK = -1  # Set by torchrun automatically
    cfg.ENV.WORLD_SIZE = 1
    cfg.ENV.DIST_BACKEND = 'nccl'  # 'nccl' for GPU, 'gloo' for CPU
    cfg.ENV.DIST_URL = 'env://'  # Use environment variables set by torchrun
    # Print detailed information
    # E.g. trainer, dataset, and backbone
    cfg.ENV.VERBOSE = True
    # Name and description of the experiment 
    cfg.ENV.NAME = ""
    cfg.ENV.DESCRIPTION = ""
    cfg.ENV.USE_WANDB = False
    # Container for arbitrary runtime/env-specific options
    cfg.ENV.EXTRA = CN(new_allowed=True)
    
    cfg.ENV.USE_WANDB = True
    cfg.ENV.WANDB = CN()
    cfg.ENV.WANDB.KEY = None
    cfg.ENV.WANDB.ENTITY = "3DVZHao"
    cfg.ENV.WANDB.PROJECT = "FLowTalker"
    cfg.ENV.WANDB.NAME = "TrainingModel"
    cfg.ENV.WANDB.NOTES = "Training as baseline."
    cfg.ENV.WANDB.TAGS = "Baseline"
    cfg.ENV.WANDB.MODE = "online"

    ###########################
    # Input
    ###########################
    cfg.INPUT = CN()
    # If True, tfm_train and tfm_test will be None
    cfg.INPUT.NO_TRANSFORM = False
    # Gaussian noise
    cfg.INPUT.GN_MEAN = 0.0
    cfg.INPUT.GN_STD = 0.15
    # RandomAugment
    cfg.INPUT.RANDAUGMENT_N = 2
    cfg.INPUT.RANDAUGMENT_M = 10

    ###########################
    # Dataset
    ###########################
    cfg.DATASET = CN()
    cfg.DATASET.NAME = ""
    cfg.DATASET.ROOT = ""   # Directory where datasets are stored
    # Percentage of validation data, set to 0 if do not want to use val data
    cfg.DATASET.VAL_PERCENT = 0.1

    # for HDTF_TFHP
    cfg.DATASET.HDTF_TFHP = CN()
    cfg.DATASET.HDTF_TFHP.LMDB = ""
    cfg.DATASET.HDTF_TFHP.COEF_STATS = "stats_train.npz"
    cfg.DATASET.HDTF_TFHP.TRAIN = "train.txt"
    cfg.DATASET.HDTF_TFHP.VAL = "val.txt"
    cfg.DATASET.HDTF_TFHP.TEST = "test.txt"
    cfg.DATASET.HDTF_TFHP.COEF_FPS = 25      # frames per second for coefficients (sequence fps)
    cfg.DATASET.HDTF_TFHP.MOTIONS = 100      # number of motions per sample
    cfg.DATASET.HDTF_TFHP.N_PREV_MOTIONS = 10   # audio sampling rate
    cfg.DATASET.HDTF_TFHP.CROP = "random"    # crop strategy
    cfg.DATASET.HDTF_TFHP.AUDIO_SR = 16000   # audio sampling rate
    cfg.DATASET.HDTF_TFHP.USE_CONTEXT_AUDIO = True  # whether to use context audio for model input
    cfg.DATASET.HDTF_TFHP.TRUNC_PROB1 = 0.3 # truncation probability for clip 1
    cfg.DATASET.HDTF_TFHP.TRUNC_PROB2 = 0.4 # truncation probability for clip 2
    cfg.DATASET.HDTF_TFHP.PAD_MODE = 'zero' # 'zero' or 'replicate'

    ###########################
    # Dataloader
    ###########################
    cfg.DATALOADER = CN()
    cfg.DATALOADER.NUM_WORKERS = 4
    # Setting for the train data-loader
    cfg.DATALOADER.TRAIN = CN()
    cfg.DATALOADER.TRAIN.BATCH_SIZE = 32

    # Setting for the test data-loader
    cfg.DATALOADER.TEST = CN()
    cfg.DATALOADER.TEST.BATCH_SIZE = 32

    ###########################
    # Model
    ###########################
    cfg.MODEL = CN()
    cfg.MODEL.NAME = ""
    cfg.MODEL.INIT_WEIGHTS = ""   # Path to model weights (for initialization)
    cfg.MODEL.AUDIO_MODEL = 'wav2vec2'
    cfg.MODEL.AUDIO_DIM = 128

    cfg.MODEL.MLP = CN()
    cfg.MODEL.MLP.INPUT_DIM = 784
    cfg.MODEL.MLP.HIDDEN_DIM = [128, 64]
    cfg.MODEL.MLP.OUTPUT_DIM = 10

    

    
    
    # Definition of embedding layers
    cfg.MODEL.HEAD = CN()
    # If none, do not construct embedding layers, the
    # backbone's output will be passed to the classifier
    cfg.MODEL.HEAD.NAME = ""
    # Structure of hidden layers (a list), e.g. [512, 512]
    # If undefined, no embedding layer will be constructed
    cfg.MODEL.HEAD.HIDDEN_LAYERS = ()
    cfg.MODEL.HEAD.ACTIVATION = "relu"
    cfg.MODEL.HEAD.BN = True
    cfg.MODEL.HEAD.DROPOUT = 0.0

    
    
    cfg.MODEL.HEAD.USE_INDICATOR = False  # Use indicator for padding frames

    # optional head type according to different input
    cfg.MODEL.HEAD.ROT_REPR = 'aa'
    cfg.MODEL.HEAD.NO_HEAD_POSE = False

    # align mask width for transformer decoder
    cfg.MODEL.HEAD.ALIGN_MASK_WIDTH = 0
    # learnable positional encoding
    cfg.MODEL.HEAD.USE_LEARNABLE_PE = True

    cfg.MODEL.BACKBONE = CN()
    cfg.MODEL.BACKBONE.NAME = ""
    cfg.MODEL.BACKBONE.IN_DIM = 15069
    cfg.MODEL.BACKBONE.HIDDEN_SIZE = 1024
    cfg.MODEL.BACKBONE.NUM_HIDDEN_LAYERS = 6
    cfg.MODEL.BACKBONE.NUM_ATTENTION_HEADS = 8
    cfg.MODEL.BACKBONE.INTERMEDIATE_SIZE = 1536
    cfg.MODEL.BACKBONE.WINDOW_SIZE = 1
    # for VQ-VAE config
    cfg.MODEL.BACKBONE.QUANT_FACTOR = 0
    cfg.MODEL.BACKBONE.FACE_QUAN_NUM = 16
    cfg.MODEL.BACKBONE.NEG = 0.2
    cfg.MODEL.BACKBONE.INAFFINE = False

    # for diffusion model
    cfg.MODEL.BACKBONE.N_STEPS = 1000
    cfg.MODEL.BACKBONE.DIFF_SCHEDULE = 'cosine'  # linear, cosine, quadratic, sigmoid

    # classifier-free guidance
    cfg.MODEL.CFG_MODE = 'incremental' # 'full', 'incremental', 'none'
    cfg.MODEL.GUIDING_CONDITIONS = 'audio,style'

    cfg.MODEL.TAIL = CN()
    cfg.MODEL.TAIL.NAME = ""
    cfg.MODEL.TAIL.NUM_HIDDEN_LAYERS = 4
    cfg.MODEL.TAIL.MLP_RATIO = 4
    cfg.MODEL.TAIL.TYARGET = "sample"  # for diffusion model, either "sample" or "noise"

    ###########################
    # Optimization
    ###########################
    cfg.OPTIM = CN()
    
    # Optimizer
    ## adam
    cfg.OPTIM.NAME = "adam"
    cfg.OPTIM.LR = 0.001
    cfg.OPTIM.WEIGHT_DECAY = 5e-4
    cfg.OPTIM.ADAM_BETA1 = 0.9
    cfg.OPTIM.ADAM_BETA2 = 0.999
    
    # unkonwn
    cfg.OPTIM.STEP_LR = True
    cfg.OPTIM.ADAPTIVE_LR = False
    cfg.OPTIM.FACTOR = 0.3
    cfg.OPTIM.MOMENTUM = 0.9
    
    # sgd
    cfg.OPTIM.SGD_DAMPNING = 0
    cfg.OPTIM.SGD_NESTEROV = True

    cfg.OPTIM.RMSPROP_ALPHA = 0.99
    # The following also apply to other
    # adaptive optimizers like adamw
    
    # STAGED_LR allows different layers to have
    # different lr, e.g. pre-trained base layers
    # can be assigned a smaller lr than the new
    # classification layer
    cfg.OPTIM.STAGED_LR = False
    cfg.OPTIM.NEW_LAYERS = ()
    cfg.OPTIM.BASE_LR_MULT = 0.1
    # Learning rate update frequency (in iterations/steps)
    # Set to 1 to update every step (default)
    # Set to N to update every N steps (useful for iteration-based training)
    cfg.OPTIM.LR_UPDATE_FREQ = 1

    # Learning rate scheduler
    ## training settings
    cfg.OPTIM.LR_SCHEDULER = "single_step"
    cfg.OPTIM.STEP_SIZE = 20
    cfg.OPTIM.GAMMA = 0.5  # Multiplicative factor of learning rate decay for 'single/multi step'

    ## warmup settings
    ### Set WARMUP_EPOCH/WARMUP_ITERS larger than 0 to activate warmup training
    cfg.OPTIM.WARMUP_EPOCHS = -1
    cfg.OPTIM.WARMUP_ITERS = -1

    cfg.OPTIM.WARMUP_TYPE = "linear"
    cfg.OPTIM.WARMUP_CONS_LR = 1e-5 # Constant learning rate when type=constant
    cfg.OPTIM.WARMUP_MIN_LR = 1e-5 # Minimum learning rate when type=linear
    ### Recount epoch for the next scheduler (last_epoch=-1)
    ### Otherwise last_epoch=warmup_epoch
    cfg.OPTIM.WARMUP_RECOUNT = True
    cfg.OPTIM.WARMUP_MULTIPLIER = 0 # for GradualWarmupScheduler
    cfg.OPTIM.MIN_LR_RATIO = 2e-6 # for gradualThenDecay

    ###########################
    # Trainer specifics
    ###########################
    cfg.TRAINER = CN()
    cfg.TRAINER.NAME = ""

    ###########################
    # Train
    ###########################
    cfg.TRAIN = CN()
    cfg.TRAIN.USE_SGD = False
    cfg.TRAIN.SYNC_BN = False  # adopt sync_bn or not
    
    ## training settings
    cfg.TRAIN.USE_ITERS = False
    cfg.TRAIN.START_EPOCH = 0
    cfg.TRAIN.MAX_EPOCHS = 50
    cfg.TRAIN.MAX_ITERS = 10000
    # How often (batch) to print training information
    cfg.TRAIN.PRINT_FREQ = 10
    # How often (epoch) to save model during training
    # Set to 0 or negative value to only save the last one
    cfg.TRAIN.SAVE_FREQ = 0

    # Whether to perform evaluation during training
    cfg.TRAIN.EVALUATE = True
    cfg.TRAIN.EVAL_FREQ = 10
    cfg.TRAIN.RENDER_FREQ = 10

    ###########################
    # Test
    ###########################
    cfg.TEST = CN()
    # If NO_TEST=True, no testing will be conducted
    cfg.TEST.NO_TEST = False
    # Use test or val set for FINAL evaluation
    cfg.TEST.SPLIT = "test"
    # Which model to test after training (last_step or best_val)
    # If best_val, evaluation is done every epoch (if val data
    # is unavailable, test data will be used)
    cfg.TEST.FINAL_MODEL = "last_step"

    cfg.EVALUATE = CN()
    cfg.EVALUATE.EVALUATOR = "TalkerEvaluator"
    cfg.EVALUATE.LOAD_RENDER = False
    cfg.EVALUATE.SAVE_COEF = False
    cfg.EVALUATE.TARGET = "sample" # foe diffusion model, either
    cfg.EVALUATE.NO_CONSTRAIN_PREV = False
    
    cfg.EVALUATE.LOSS = CN()
    cfg.EVALUATE.LOSS.NAME = "L2Loss"
    cfg.EVALUATE.LOSS.CONTRASTIVE = CN()
    cfg.EVALUATE.LOSS.CONTRASTIVE.TEMPRATURE = 0.1
    cfg.EVALUATE.LOSS.GEOMETRIC = CN()
    cfg.EVALUATE.LOSS.GEOMETRIC.W_VERTEX = 2e6  # weight of the vertex loss
    cfg.EVALUATE.LOSS.GEOMETRIC.W_VELOCITY = 1e7  # weight of the velocity loss
    cfg.EVALUATE.LOSS.GEOMETRIC.W_SMOOTH = 1e5   # weight of the vertex acceleration regularization
    cfg.EVALUATE.LOSS.GEOMETRIC.HEAD = CN()
    cfg.EVALUATE.LOSS.GEOMETRIC.HEAD.W_ANGLE = 0.05  # weight of the head angle loss
    cfg.EVALUATE.LOSS.GEOMETRIC.HEAD.W_VELOCITY = 5.0 # weight of the head angular velocity loss
    cfg.EVALUATE.LOSS.GEOMETRIC.HEAD.W_SMOOTH = 0.5  # weight of the head angular acceleration regularization
    cfg.EVALUATE.LOSS.GEOMETRIC.HEAD.W_TRANS = 0.5  # weight of the head constraint during window transition

    cfg.EVALUATE.TDMM = CN()
    cfg.EVALUATE.TDMM.FLAME = CN()
    cfg.EVALUATE.TDMM.FLAME.ROOT = "pretrained/FLAME"

    cfg.EVALUATE.RENDER = CN()
    cfg.EVALUATE.RENDER.NAME = "PyMeshRenderer"
    cfg.EVALUATE.RENDER.REND_SIZE = (640, 640)
    cfg.EVALUATE.RENDER.BLACK_BG = False
    # OP
    self.cfg = cfg