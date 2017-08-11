from __future__ import absolute_import
from six.moves import range

import os

import nnabla as nn
import nnabla.logger as logger
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S

from args import get_args
from mnist_data import data_iterator_mnist

from mnist_model import mlp, mlp_dni, cnn, cnn_dni, \
    ce_loss, se_loss, categorical_error

def train():
    args = get_args()

    # Get context.
    from nnabla.contrib.context import extension_context
    extension_module = args.context
    if args.context is None:
        extension_module = 'cpu'
    logger.info("Running in %s" % extension_module)
    ctx = extension_context(extension_module, device_id=args.device_id)
    nn.set_default_context(ctx)

    # Train
    image = nn.Variable([args.batch_size, 1, 28, 28])
    label = nn.Variable([args.batch_size, 1])
    h_d, h_copy, pred, g_pred, g_label = cnn_dni(image, y=label)
    loss_ce = ce_loss(pred, label)          # loss of a problem at hand
    loss_se = se_loss(g_pred, g_label)  # gradient synthesizer loss

    # Test
    vimage = nn.Variable([args.batch_size, 1, 28, 28])
    vlabel = nn.Variable([args.batch_size, 1])
    vpred = cnn(vimage, test=True)

    # Solver
    solver = S.Adam(args.learning_rate)
    with nn.parameter_scope("ref"):
        solver.set_parameters(nn.get_parameters())
    solver_gs = S.Adam(args.learning_rate)
    with nn.parameter_scope("gs"):
        solver_gs.set_parameters(nn.get_parameters())

    # Monitor
    from nnabla.monitor import Monitor, MonitorSeries, MonitorTimeElapsed
    monitor = Monitor(args.monitor_path)
    monitor_loss = MonitorSeries("Training loss", monitor, interval=10)
    monitor_err = MonitorSeries("Training error", monitor, interval=10)
    monitor_time = MonitorTimeElapsed("Training time", monitor, interval=100)
    monitor_verr = MonitorSeries("Test error", monitor, interval=10)

    # DataIterator
    data = data_iterator_mnist(args.batch_size, True)
    vdata = data_iterator_mnist(args.batch_size, False)

    # Training loop
    for i in range(args.max_iter):
        if i % args.val_interval == 0:
            # Validation
            ve = 0.0
            for j in range(args.val_iter):
                vimage.d, vlabel.d = vdata.next()
                vpred.forward(clear_buffer=False)
                ve += categorical_error(vpred.d, vlabel.d)
            monitor_verr.add(i, ve / args.val_iter)
        if i % args.model_save_interval == 0:
            nn.save_parameters(os.path.join(
                args.model_save_path, 'params_%06d.h5' % i))

        # Training
        image.d, label.d = data.next()
        solver.zero_grad()
        solver_gs.zero_grad()

        ## forward
        h_d.forward(clear_no_need_grad=False)
        loss_ce.forward(clear_no_need_grad=False)
        loss_se.forward(clear_no_need_grad=False)

        ## backward
        loss_ce.backward(clear_buffer=False)
        h_d.backward(clear_buffer=False)
        loss_se.backward(clear_buffer=False)

        ## update
        solver.weight_decay(args.weight_decay)
        solver.update()
        solver_gs.weight_decay(args.weight_decay)
        solver_gs.update()

        ## monitor
        e = categorical_error(pred.d, label.d)
        monitor_loss.add(i, loss_ce.d.copy())
        monitor_err.add(i, e)
        monitor_time.add(i)

    nn.save_parameters(os.path.join(args.model_save_path,
                                    'params_%06d.h5' % args.max_iter))


if __name__ == '__main__':
    train()
