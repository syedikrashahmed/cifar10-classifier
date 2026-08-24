EXPERIMENTS = {

    "mlp_baseline": {
        "model": "mlp",
        "model_type": "cnn",
        "augment": False,
        "batch_size": 64,
        "optimizer": "adam",
        "lr": 0.001,
        "weight_decay": 0.0,
        "epochs": 10,
        "seed": 42,
    },

    "cnn_baseline": {
        "model": "cnn",
        "model_type": "cnn",
        "augment": False,
        "batch_size": 64,
        "optimizer": "adam",
        "lr": 0.001,
        "weight_decay": 0.0,
        "epochs": 10,
        "seed": 42,
    },

    "cnn_augmentation": {
        "model": "cnn",
        "model_type": "cnn",
        "augment": True,
        "batch_size": 64,
        "optimizer": "adam",
        "lr": 0.001,
        "weight_decay": 0.0,
        "epochs": 10,
        "seed": 42,
    },

    "cnn_best": {
        "model": "cnn",
        "model_type": "cnn",
        "augment": True,
        "batch_size": 32,
        "optimizer": "adam",
        "lr": 0.001,
        "weight_decay": 0.0,
        "epochs": 20,
        "seed": 42,
    },

    "resnet_feature": {
        "model": "resnet_feature",
        "model_type": "resnet",
        "augment": True,
        "batch_size": 64,
        "optimizer": "adam",
        "lr": 0.001,
        "weight_decay": 0.0,
        "epochs": 10,
        "seed": 42,
    },

    "resnet_finetune": {
        "model": "resnet_finetune",
        "model_type": "resnet",
        "augment": True,
        "batch_size": 64,
        "optimizer": "adam",
        "lr": 0.001,
        "weight_decay": 0.0,
        "epochs": 10,
        "seed": 42,
    },
}