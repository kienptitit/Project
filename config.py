import os


class CFG:
    continue_training = False  # Set to True If want to continue training from specific epoch

    Batch_size = 16
    num_epochs = 25
    sub_epochs = 8
    in_features = 1024
    pos_embed_dim = 128
    coupling_layers = 8
    clamp_alpha = 1.9
    lr = 0.001
    lr_cosine = True
    lr_decay_rate = 0.1
    lr_decay_epochs = [50, 75, 90]

    flow_arch = 'conditional_flow_model'
    checkpoint = None
    device = 'cuda'
    snippets = 32
    normal_sub_epoch = 2
    focal_weighting = False
    normalizer = 10.0
    pos_beta = 0.4
    margin_abnormal_negative = 0.2 / normalizer
    margin_abnormal_positive = 0.1 / normalizer
    bgspp_lambda = 1.0
    lr_warm = True
    lr_warm_epochs = 2
    save_result = True
    save_boundary = True
    mode_loss = 1

    result_path = r"E:\2023\NaverProject\LastCodingProject\Result_constrastive_stable_boundary"  # Path to save visualization boundary
    log_path = r"E:\2023\NaverProject\LastCodingProject\logs\Train_Constrastive_Stable_Boundary"  # Path to save log model
    record_train_saved = r"train.pickle"
    record_test_saved = r"test.pickle"
    model_saved_path = r"model.pt"
    label_train_path = r"E:\2023\NaverProject\LastCodingProject\Binary_file\label_train_flow.pt"
    label_test_path = r"E:\Python test Work\ConAno\labels\UCF_test.txt"  # Label_test_path
    test_path = r'E:\2023\NaverProject\LastCodingProject\Binary_file\X_test_flow.npy'
    train_path = r"E:\2023\NaverProject\LastCodingProject\Binary_file\X_train_flow.npy"

    update_boundary_frequency = 3
    if not os.path.exists(result_path):
        os.mkdir(result_path)
