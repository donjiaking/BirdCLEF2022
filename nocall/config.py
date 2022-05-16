class CFG:
    # Preprocess #
    sample_rate = 32000
    segment = sample_rate * 5

    # MelSpectrogram #
    n_fft = 2048
    win_length = 2048
    hop_length = 512 
    n_mels = 64
    fmin = 16
    fmax = 16386
    power = 2

    # Model #
    model_name = 'resnext50_32x4d'
    target_size = 2
    pretrained = True

    # Training #
    num_workers = 4
    epochs = 15
    lr = 1e-4
    print_freq = 50  # print result every 50 batches    
    batch_size = 16

    # Evaluation #
    target_col = 'hasbird'
    
    # CosineAnnealingWarmRestarts #
    scheduler = 'CosineAnnealingWarmRestarts'
    T_0 = 5
    min_lr = 5e-7

    # Optional #
    n_fold = 5
    weight_decay = 1e-5
    max_grad_norm = 100
    # seed = 7    
    
    # Path #
    root_path = '../input/ff1010bird/'
    wav_path = root_path + 'ff1010bird_wav/'
    meta_path = root_path + 'ff1010bird_metadata_2018.csv'

