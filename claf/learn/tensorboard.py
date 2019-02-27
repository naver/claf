
import os

from tensorboardX import SummaryWriter


class TensorBoard:
    """ TensorBoard Wrapper for Pytorch """

    def __init__(self, log_dir):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.writer = SummaryWriter(log_dir=log_dir)

    def scalar_summaries(self, step, summary):
        for tag, value in summary.items():
            self.scalar_summary(step, tag, value)

    def scalar_summary(self, step, tag, value):
        """Log a scalar variable."""
        self.writer.add_scalar(tag, value, step)

    def image_summary(self, tag, images, step):
        """Log a list of images."""
        raise NotImplementedError()

    def embedding_summary(self, features, metadata=None, label_img=None):
        raise NotImplementedError()

    def histogram_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""
        raise NotImplementedError()

    def graph_summary(self, model, input_to_model=None):
        raise NotImplementedError()
