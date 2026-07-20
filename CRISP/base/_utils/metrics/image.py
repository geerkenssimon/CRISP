# build-in modules
from math import exp

# third party mopdules
import torch
import torch.nn.functional as F
from torch.autograd import Variable


class SSIM(torch.nn.Module):
    """Class for SSIM computation."""

    def __init__(self, shape, device, window_size=11, size_average=True):
        super().__init__()
        self.shape = shape
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(self.window_size, self.channel).to(device)

    @staticmethod
    def gaussian(window_size: int, sigma: float) -> torch.Tensor:
        """Function computing a gaussian distribution.

        Parameters
        ----------
        window_size : int
            Window Size of the Distribution
        sigma : float
            Sigma parameter of Gaussian

        Returns
        -------
        torch.Tensor
        """
        gauss = torch.Tensor([exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window(self, window_size: int, channel) -> torch.Tensor:
        """Function for generating a Window based on a given size."""
        _1d_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2d_window = _1d_window.mm(_1d_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2d_window.expand(channel, 1, window_size, window_size).contiguous())
        return window

    def forward(self, x):
        """Forward funtion."""
        return self.distance(x)

    @staticmethod
    def _ssim(
        img1: torch.Tensor,
        img2: torch.Tensor,
        window: Variable,
        window_size: int,
        channel: int,
        size_average: bool = True,
    ) -> torch.Tensor:
        """Function for computing the SSIM between two images.

        Parameters
        ----------
        img1 : torch.Tensor
            Image 1 to be compared
        img2 : torch.Tensor
            Image 2 to be compared
        window : Variable
            Window slided over the image for comparison areas
        window_size : int
            Size of the window
        channel : int
            Channels of the images
        size_average : bool, optional
            Whether to average the images sizes, by default True

        Returns
        -------
        torch.Tensor
        """
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        c1 = 0.01**2
        c2 = 0.03**2

        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))

        if size_average:
            return ssim_map.mean()
        return ssim_map.mean(1).mean(1).mean(1)

    def distance(self, data: torch.Tensor):
        """Function for sitance computation.

        Parameters
        ----------
        data : torch.Tensor
            Data to compute the distances on
        """
        (_, _, channel, _, _) = data.size()

        if channel == self.channel and self.window.data.type() == data.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)

            if data.is_cuda:
                window = window.cuda(data.get_device())
            window = window.type_as(data)

            self.window = window
            self.channel = channel

        distances = torch.zeros((data.size(0), data.size(0))).to(data.device)
        for i, img1 in enumerate(data):
            for j, img2 in enumerate(data):
                distances[i, j] = self._ssim(img1, img2, window, self.window_size, channel, self.size_average)

        return distances