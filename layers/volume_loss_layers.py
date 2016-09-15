import caffe
import numpy as np

class SigmoidCrossEntropyLossLayer(caffe.Layer):
    """
    Binomial cross-entropy loss layer with mask.
    """

    def setup(self, bottom, top):
        # Check inputs.
        if len(bottom) != 3:
            raise Exception("Need three inputs (propagated, label, mask).")
        # Params is a python dictionary with layer parameters.
        params = eval(self.param_str)
        # Loss weight.
        self.loss_weight = params.get('loss_weight', 1.0)
        # Threshold for computing classification error
        self.thresh = 0.5

    def reshape(self, bottom, top):
        # Check input dimensions match.
        if (bottom[0].count != bottom[1].count or
            bottom[0].count != bottom[2].count):
            raise Exception("Inputs must have the same dimension.")
        # Difference is shape of inputs.
        self.diff = np.zeros_like(bottom[0].data)
        self.cost = np.zeros_like(bottom[0].data)
        self.cerr = np.zeros_like(bottom[0].data)
        # Loss outputs are scalar.
        top[0].reshape(1)  # Rebalanced loss
        top[1].reshape(1)  # Unbalanced loss
        top[2].reshape(1)  # Classification error

    def forward(self, bottom, top):
        prob  = self.sigmoid(bottom[0].data)
        label = bottom[1].data
        mask  = bottom[2].data
        # Loss weight.
        mask = mask * self.loss_weight
        # Gradient.
        self.diff[...] = mask*(prob - label)
        # Cross entropy.
        self.cost[...] = self.cross_entropy(bottom[0].data, label)
        # Classification error.
        self.cerr[...] = (mask>0)*((prob>self.thresh) != (label>self.thresh))
        # Rebalanced cost.
        top[0].data[...] = np.sum(mask*self.cost)
        # Unbalanced cost.
        top[1].data[...] = np.sum(self.cost)
        # Classification error.
        top[2].data[...] = np.sum(self.cerr)

    def backward(self, top, propagate_down, bottom):
        if propagate_down[1] or propagate_down[2]:
            raise Exception("Cannot backpropagate to label or mask inputs.")
        if propagate_down[0]:
            bottom[0].diff[...] = self.diff

    def sigmoid(self, x):
        """Numerically-stable sigmoid function."""
        ret = np.zeros_like(x)
        idx = x >= 0
        z = np.exp(-x[idx])
        ret[idx] = 1 / (1 + z)
        idx = x < 0
        z = np.exp(x[idx])
        ret[idx] = z / (1 + z)
        return ret

    def cross_entropy(self, y, t):
        """Numerically-stable binomial cross-entropy.

        Args:
            y: Prediction
            t: Target (ground truth)
        """
        return -y*(t - (y>=0)) + np.log(1 + np.exp(y - 2*y*(y>=0)))


class SoftmaxLossLayer(caffe.Layer):
    """
    Softmax cross-entropy loss layer with mask.
    """

    def setup(self, bottom, top):
        # Check inputs.
        if len(bottom) != 3:
            raise Exception("Need three inputs (propagated, label, mask).")
        # Threshold for computing classification error
        self.thresh = 0.5

    def reshape(self, bottom, top):
        # Check input dimensions match.
        if (bottom[0].count != bottom[1].count or
            bottom[0].count != bottom[2].count):
            raise Exception("Inputs must have the same dimension.")
        # Difference is shape of inputs.
        self.diff = np.zeros_like(bottom[0].data)
        self.cost = np.zeros_like(bottom[0].data)
        self.cerr = np.zeros_like(bottom[0].data)
        # Loss outputs are scalar.
        top[0].reshape(1)  # Rebalanced loss
        top[1].reshape(1)  # Unbalanced loss
        top[2].reshape(1)  # Classification error

    def forward(self, bottom, top):
        prob  = self.softmax(bottom[0].data)
        label = bottom[1].data
        mask  = bottom[2].data
        # Gradient
        self.diff[...] = mask*(prob - label)
        # Cross entropy
        self.cost[...] = self.cross_entropy(prob, label)
        # Classification error
        self.cerr[...] = (mask>0)*((prob>self.thresh) != (label>self.thresh))
        # Rebalanced cost
        top[0].data[...] = np.sum(mask*self.cost)
        # Unbalanced cost
        top[1].data[...] = np.sum(self.cost)
        # Classification error
        top[2].data[...] = np.sum(self.cerr)

    def backward(self, top, propagate_down, bottom):
        if propagate_down[1] or propagate_down[2]:
            raise Exception("Cannot backpropagate to label or mask inputs.")
        if propagate_down[0]:
            bottom[0].diff[...] = self.diff

    def softmax(self, x):
        """Numerically-stable softmax function."""
        ret = x.copy()
        # Maximum along the channel dimension.
        propmax = np.max(x, axis=-4)
        # Subtract the maximum.
        for b in range(x.shape[0]):
            for c in range(x.shape[1]):
                ret[b,c,...] -= propmax[b]
        # Exponentiate.
        ret = np.exp(ret)
        # Sum after exp. along the channel dimension.
        expsum = np.sum(ret, axis=-4)
        # Divide.
        for b in range(x.shape[0]):
            for c in range(x.shape[1]):
                ret[b,c,...] /= expsum[b]
        return ret

    def cross_entropy(self, y, t):
        """Multinomial cross-entropy.

        Args:
            y: Prediction
            t: Target (ground truth)
        """
        return -t * np.log(y)
