class CFG:
    # Pre-process #
    BACKGROUND_PATH1 = "../input/ff1010bird_nocall/nocall"
    BACKGROUND_PATH2 = "../input/train_soundscapes/nocall"
    BACKGROUND_PATH3 = "../input/aicrowd2020_noise_30sec/noise_30sec"
    gainTransition_p = 0.5
    gaussianSNR_p = 0.8

    # Melspectrogram #
    sample_rate = 32000
    n_fft = 2048
    win_length = 2048
    hop_length = 512 
    n_mels = 64
    fmin = 16
    fmax = 16386
    power = 2
    segment_train = sample_rate * 30  # 5
    segment_test = sample_rate * 5

    # Train #
    num_epochs = 20
    lr = 1e-3
    weight_decay = 1e-6
    batch_size = 16
    val_batch_size = 1
    print_feq = 50

    n_classes = 152
    backbone = 'tf_efficientnetv2_s_in21k'  # 'resnext50' 'eca_nfnet_l0'
    pretrained = True
    mix_beta = 1

    # Input Data #
    root_path = "../input/birdclef-2022/"
    input_path = root_path + 'train_audio/'
    test_audio_path = '../input/birdclef-2022/test_soundscapes/'
    model_out_path = './models/'
    min_rating = 2.0

    # Binary Threshold #
    binary_th = 0.3

    
