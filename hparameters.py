hparameters = {
    "num_layers":4,
    "d_model":128,
    "dff":512,
    "num_heads":8,
    "batch_size":64,
    "max_sentence_len":15,  # 可接受的最大句长
    "training_dataset":'./deu.txt', # 训练文本路径
    "log_every_step":50,
    "input_vocab_size":8500,
    "target_vocab_size":8000,
    "dropout_rate":0.1,
    "checkpoint_path":"./output/checkpoints/",
    "maximum_position_encoding":10000,
    "training_epochs" : 1
}