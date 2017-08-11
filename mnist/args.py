def get_args(monitor_path='tmp.monitor', max_iter=10000, model_save_path=None, learning_rate=1e-3, batch_size=32, weight_decay=0):
    """
    Get command line arguments.

    Arguments set the default values of command line arguments.
    """
    import argparse
    import os
    if model_save_path is None:
        model_save_path = monitor_path
    parser = argparse.ArgumentParser(
        description="Examples on MNIST dataset. The following help shared among examples in this folder. Some argumetns are valid or invalid in some examples.")
    parser.add_argument("--batch-size", "-b", type=int, default=batch_size)
    parser.add_argument("--learning-rate", "-l",
                        type=float, default=learning_rate)
    parser.add_argument("--monitor-path", "-m",
                        type=str, default=monitor_path,
                        help='Path monitoring logs saved.')
    parser.add_argument("--max-iter", "-i", type=int, default=max_iter,
                        help='Max iteration of training.')
    parser.add_argument("--val-interval", "-v", type=int, default=100,
                        help='Validation interval.')
    parser.add_argument("--val-iter", "-j", type=int, default=10,
                        help='Each validation runs `val_iter mini-batch iteration.')
    parser.add_argument("--weight-decay", "-w",
                        type=float, default=weight_decay,
                        help='Weight decay factor of SGD update.')
    parser.add_argument("--device-id", "-d", type=int, default=0,
                        help='Device ID the training run on. This is only valid if you specify `-c cuda.cudnn`.')
    parser.add_argument("--model-save-interval", "-s", type=int, default=1000,
                        help='The interval of saving model parameters.')
    parser.add_argument("--model-save-path", "-o",
                        type=str, default=model_save_path,
                        help='Path the model parameters saved.')
    parser.add_argument("--net", "-n", type=str,
                        default='lenet',
                        help="Neural network architecure type (used only in classification*.py).\n  classification.py: ('lenet'|'resnet'),  classification_bnn.py: ('bincon'|'binnet'|'bwn'|'bwn'|'bincon_resnet'|'binnet_resnet'|'bwn_resnet')")
    parser.add_argument('--context', '-c', type=str,
                        default=None, help="Extension modules. ex) 'cpu', 'cuda.cudnn'.")
    args = parser.parse_args()
    if not os.path.isdir(args.model_save_path):
        os.makedirs(args.model_save_path)
    return args
