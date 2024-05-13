"""Color Equivariant Convolutional Layer."""

import math
import typing
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter


def _trans_input_filter_hsv(weights, out_rotations, hue_shift, sat_shift) -> torch.Tensor:
    """Apply Hue / Sat shift to filters of the input layer

    Args:
        weights: float32, input filter of size [c_out, c_in (3), in_rot (1), k, k]
        out_rotations: int, number of rotations applied to filter
        hue_shift: Bool, signaling whether or not to perform a hue shift on the filter
        sat_shift: Bool, signaling whether or not to perform a saturation shift on the filter
    """
    if not hue_shift and not sat_shift:
        raise ValueError("Flag hue_shift, sat_shift or both -> Cant be left empty in hsv_space")
    if sat_shift:
        neg_sats = out_rotations // 2
        pos_sats = neg_sats - 1 + out_rotations % 2
        saturation_scales = torch.concat((torch.linspace(-1, 0, neg_sats + 1)[:-1],
                                    torch.tensor([0]),
                                    torch.linspace(0, 1, pos_sats + 1)[1:])).type(torch.float32)

    # [c_out, 1, c_in (3), in_rot (1), k, k]
    # Where dim=1 is added for stacking every hue_shift / sat_shift combination
    weights_reshaped = weights.unsqueeze(1)

    hue_shifted_filters = []
    final_weights = []

    if hue_shift:
        # Loop over all rotations
        for i in range(out_rotations):
            transformed_weights = weights_reshaped.clone()
            # For rotation i of the group -> with an angle i/out_rotations
            # Add this hue angle to the current weights
            temp = transformed_weights[:, :, 0, :, :, :] + (i / out_rotations * 2*np.pi)
            # Restrict the values between 0 - 2pi for the hue using modulo
            transformed_weights[:, :, 0, :, :, :] = torch.remainder(temp, 2*np.pi)
            hue_shifted_filters.append(transformed_weights)
    else:
        # Didn't Hue shift
        hue_shifted_filters.append(weights_reshaped.clone())

    if sat_shift:
        for filter in hue_shifted_filters:
            for i in range(out_rotations):
                transformed_weights = filter.clone()
                # A saturation shift on number of sat shifts = i on the interval between 
                # -1 and 1. Add this shift to the saturation channel
                temp = transformed_weights[:, :, 1, :, :, :] + saturation_scales[i]
                # constraint the value of the sat channel of the filter between 0 - 1
                transformed_weights[:, :, 1, :, :, :] = torch.clip(temp, min=0., max=1.)
                final_weights.append(transformed_weights)
    else:
        # Didn't Sat shift
        final_weights = hue_shifted_filters

    tw = torch.concat(final_weights, dim=1)
    
    return tw.contiguous()


def _trans_input_filter_repeat(weights, out_rotations, hue_shift, sat_shift) -> torch.Tensor:
    """Stack the same weight filter out_rotation number of times (shift either hue or sat),
    or out_rotations^2 times in the case of both hue and sat shift

    Args:
      weights: float32, input filter of size [c_out, 3 (c_in), 1, k, k]
      out_rotations: int, number of rotations (Hue shifts)
      hue_shift: Bool, signaling whether or not to perform a hue shift 
      sat_shift: Bool, signaling whether or not to perform a saturation shift
    """
    # If either hue shift or sat shift is required
    # stack only out_rotation time the input filer
    if hue_shift != sat_shift:
        # Reshape to [c_out, 1, 3 (c_in), 1, k, k]
        weights = weights.unsqueeze(1)
        # [c_out, rotations, c_in (3), 1, k, k]
        tw = weights.repeat(1,out_rotations,1,1,1,1)
    # If both hue and sat shift are required
    # then we need out_rotations * out_rotations number of filters
    elif hue_shift and sat_shift:
        # Reshape to [c_out, 1, 3 (c_in), 1, k, k]
        weights = weights.unsqueeze(1)
        # [c_out, out_rotations*rotations, c_in (3), 1, k, k]
        tw = weights.repeat(1,out_rotations*out_rotations,1,1,1,1)
    else:
        raise ValueError("For an input image shift, specify either hue/sat shift or both")

    return tw.contiguous()


def _shifted_img_stack(imgs, out_rotations, hue_shift, sat_shift):
    """Stack the same weight filter out_rotation number of times

    Args:
      imgs: float32, input image of size [Batch, channels, H, W]
      out_rotations: int, number of rotations (Hue / Sat shifts) applied to the input image
      hue_shift: Bool, signaling whether or not to perform a hue shift 
      sat_shift: Bool, signaling whether or not to perform a saturation shift  
    """
    imgs_stacked = []

    # Create sat shifts between -1 and 1 depending on number of "out_rotations" (shifts)
    if sat_shift:
        neg_sats = out_rotations // 2
        pos_sats = neg_sats - 1 + out_rotations % 2
        sat_shifts = np.append(np.linspace(-1, 0, neg_sats+1)[:-1], np.linspace(0, 1, pos_sats+1))

    for i in range(out_rotations):
        imgs_cloned = imgs.clone()

        if hue_shift:
            # For rotation i of the group -> with an angle: i/out_rotations * 2 pi
            # Add this hue angle to the input image
            temp = imgs[:, 0:1, :, :] + (i / out_rotations * 2*np.pi)
            # Restrict the values between 0 - 2pi for the hue using modulo
            temp = torch.remainder(temp, 2*np.pi)
            imgs_cloned[:, 0:1, :, :] = temp
        if sat_shift:
            # Add the corresponding sat shift to the input image
            temp = imgs[:, 1:2, :, :] + sat_shifts[i]
            # Restrict the sat channel to fall within the required 0-1 range
            temp = torch.clip(temp, min=0, max=1)
            imgs_cloned[:, 1:2, :, :] = temp

        imgs_stacked.append(imgs_cloned)

    # Concatenate the hue / sat shifted images to create the final image stack
    hue_shifted_img_stack = torch.cat(imgs_stacked, dim=1)

    return hue_shifted_img_stack


def _get_hue_rotation_matrix(rotations: int) -> torch.Tensor:
    """Returns a 3x3 hue rotation matrix.

    Rotates a 3D point by 360/rotations degrees along the diagonal.

    Args:
      rotations: int, number of rotations
    """

    assert rotations > 0, "Number of rotations must be positive."

    # Constants in rotation matrix
    cos = math.cos(2 * math.pi / rotations)
    sin = math.sin(2 * math.pi / rotations)
    const_a = 1 / 3 * (1.0 - cos)
    const_b = math.sqrt(1 / 3) * sin

    # Rotation matrix
    return torch.tensor(
        [
            [cos + const_a, const_a - const_b, const_a + const_b],
            [const_a + const_b, cos + const_a, const_a - const_b],
            [const_a - const_b, const_a + const_b, cos + const_a],
        ],
        dtype=torch.float32,
    )


def _get_lab_rotation_matrix(rotations: int) -> torch.Tensor:
    """Returns a 3x3 hue rotation matrix.

    Rotates a 3D point by 360/rotations degrees along the diagonal.

    Args:
      rotations: int, number of rotations
    """

    assert rotations > 0, "Number of rotations must be positive."

    # Angle to shift each image by
    # By using the matrix power we do multiple rotations by aply this matrix
    # multiple times
    angle_delta = 2 * math.pi / rotations

    # Rotation matrix
    return torch.tensor(
        [
            [1, 0, 0],
            [0, math.cos(angle_delta), -math.sin(angle_delta)],
            [0, math.sin(angle_delta), math.cos(angle_delta)]
        ], 
        dtype=torch.float32)


def _get_hsv_saturation_matrix(saturations: int) -> torch.Tensor:
    """
    Returns a [saturations, 1, 1] saturation matrix.
    Saturation assumed to be in [0,1].

    Args:
      saturations: int, number of saturation shifts
    """

    assert saturations > 0, "Number of saturation shifts must be positive."

    # Scalar to add each saturation value in image by
    neg_sats = saturations // 2
    pos_sats = neg_sats - 1 + saturations % 2

    # In case of even saturations, consider 0 to be positive
    saturation_scales = torch.concat((torch.linspace(-1, 0, neg_sats + 1)[:-1],
                                      torch.tensor([0]),
                                      torch.linspace(0, 1, pos_sats + 1)[1:])).type(torch.float32)

    return saturation_scales[:, None, None]


def _trans_input_filter(weights, rotations, rotation_matrix) -> torch.Tensor:
    """Apply linear transformation to filter.

    Args:
      weights: float32, input filter of size [c_out, 3 (c_in), 1, k, k]
      rotations: int, number of rotations applied to filter
      rotation_matrix: float32, rotation matrix of size [3, 3]
    """

    # Flatten weights tensor.
    weights_flat = weights.permute(2, 1, 0, 3, 4)  # [1, 3, c_out, k, k]
    weights_shape = weights_flat.shape
    weights_flat = weights_flat.reshape((1, 3, -1))  # [1, 3, c_out*k*k]

    # Construct full transformation matrix.
    rotation_matrix = torch.stack(
        [torch.matrix_power(rotation_matrix, i) for i in range(rotations)], dim=0
    )

    # Apply transformation to weights.
    # [rotations, 3, 3] * [1, 3, c_out*k*k] --> [rotations, 3, c_out*k*k]
    transformed_weights = torch.matmul(rotation_matrix, weights_flat)
    # [rotations, 1, c_in (3), c_out, k, k]
    transformed_weights = transformed_weights.view((rotations,) + weights_shape)
    # [c_out, rotations, c_in (3), 1, k, k]
    tw = transformed_weights.permute(3, 0, 2, 1, 4, 5)

    return tw.contiguous()


def _trans_hidden_filter(weights: torch.Tensor, rotations: int, hue_shift, sat_shift) -> torch.Tensor:
    """Perform cyclic permutation on hidden layer filter parameters."""

    if hue_shift and sat_shift:
        # Apply rotations * rotations rolls (for hue and sat shifts respectively)
        rotations_squared = rotations * rotations

        # Create placeholder for output tensor
        w_shape = weights.shape
        transformed_weights = torch.zeros(
            ((w_shape[0],) + (rotations_squared,) + w_shape[1:]), device=weights.device
        )
        # Apply cyclic permutation on output tensor

        for i in range(rotations_squared):
            transformed_weights[:, i, :, :, :, :] = torch.roll(weights, shifts=i, dims=2)
    else:
        # Only apply a single shift = # rotations amount of rolls

        # Create placeholder for output tensor
        w_shape = weights.shape
        transformed_weights = torch.zeros(
            ((w_shape[0],) + (rotations,) + w_shape[1:]), device=weights.device
        )

        # Apply cyclic permutation on output tensor
        for i in range(rotations):
            transformed_weights[:, i, :, :, :, :] = torch.roll(weights, shifts=i, dims=2)

    return transformed_weights


class CEConv2d(nn.Conv2d):
    """
    Applies a Color Equivariant convolution over an input signal composed of several
    input planes.


    Args:
        in_rotations (int): Number of input rotations: 1 for input layer, >1 for
            hidden layers.
        out_rotations (int): Number of output rotations.
        in_channels (int): Number of input channels.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int or tuple): Size of the convolving kernel.
        learnable (bool): If True, the transformation matrix is learnable.
        separable (bool): If True, the convolution is separable.
        kernel_size (int or tuple): Size of the convolving kernel.
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int, tuple or str, optional): Padding added to all four sides of
            the input. Default: 0
        padding_mode (str, optional): ``'zeros'``, ``'reflect'``,
            ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
        bias (bool, optional): If ``True``, adds a learnable bias to the
            output. Default: ``True``
    """

    def __init__(
        self,
        in_rotations: int,
        out_rotations: int,
        in_channels: int,
        out_channels: int,
        kernel_size: typing.Union[int, typing.Tuple[int, int]],
        learnable: bool = False,
        separable: bool = True,
        lab_space: bool = False,
        hsv_space: bool = False,
        shift_img: bool = True,
        sat_shift: bool = False,
        hue_shift: bool = False,
        **kwargs
    ) -> None:
        self.in_rotations = in_rotations
        self.out_rotations = out_rotations
        self.separable = separable
        self.lab_space = lab_space
        self.hsv_space = hsv_space
        self.shift_img = shift_img
        self.sat_shift = sat_shift
        self.hue_shift = hue_shift
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)      

        # Turn off seperable when dealing with 2 shifts
        if hue_shift and sat_shift:
            raise ValueError("SORRY BOBS 2 SHIFT NOT IMPLEMENTED YET!")
            separable = False
            self.separable = separable

        # Initialize transformation matrix and weights.
        if in_rotations == 1:
            if self.hsv_space:
                self.weight = Parameter(
                    torch.Tensor(out_channels, in_channels, 1, *self.kernel_size)
                )
            else:
                if learnable:
                    init = (
                        torch.rand((3, 3)) * 2.0 / 3 - (1.0 / 3)  
                    )
                elif lab_space:
                    init = _get_lab_rotation_matrix(out_rotations)
                else:
                    # For RGB space
                    init = _get_hue_rotation_matrix(out_rotations)

                self.transformation_matrix = Parameter(init, requires_grad=learnable)
                self.weight = Parameter(
                    torch.Tensor(out_channels, in_channels, 1, *self.kernel_size)
                )
        else:
            if separable:
                if in_rotations > 1:
                    self.weight = Parameter(
                        # torch.Tensor(out_channels, 1, 1, *self.kernel_size)
                        torch.Tensor(out_channels, in_channels, 1, *self.kernel_size)
                    )
                    self.pointwise_weight = Parameter(
                        torch.Tensor(out_channels, in_channels, self.in_rotations, 1, 1)
                    )
            else:
                self.weight = Parameter(
                    torch.Tensor(
                        out_channels, in_channels, self.in_rotations, *self.kernel_size
                    )
                )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters."""

        # Compute standard deviation for weight initialization.
        n = self.in_channels * self.in_rotations * np.prod(self.kernel_size)
        stdv = 1.0 / math.sqrt(n)

        # Initialize weights.
        self.weight.data.uniform_(-stdv, stdv)
        if hasattr(self, "pointwise_weight"):
            self.pointwise_weight.data.uniform_(-stdv, stdv)

        # Initialize bias.
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Compute full filter weights.
        if self.in_rotations == 1:
            
            if self.hsv_space:
                # Apply hue / sat shift to input image HSV
                if self.shift_img:
                    tw = _trans_input_filter_repeat(self.weight, self.out_rotations, self.hue_shift, self.sat_shift)
                # Apply shifts to input layer filters HSV
                else:
                    tw = _trans_input_filter_hsv(self.weight, self.out_rotations, self.hue_shift, self.sat_shift)
            # Apply rotation to input layer filter. (RGB)
            else:
                tw = _trans_input_filter(
                    self.weight, self.out_rotations, self.transformation_matrix
                )
        else:
            # Apply cyclic permutation to hidden layer filter.
            if self.separable:
                weight = torch.mul(self.pointwise_weight, self.weight)
            else:
                weight = self.weight
            tw = _trans_hidden_filter(weight, self.out_rotations, self.hue_shift, self.sat_shift)

        # if performing 2 shifts
        if self.hue_shift and self.sat_shift:
            tw_shape = (
                self.out_channels * self.out_rotations * self.out_rotations,
                self.in_channels * self.in_rotations,
                *self.kernel_size,
            )
            tw = tw.view(tw_shape)
        else:
            tw_shape = (
                self.out_channels * self.out_rotations,
                self.in_channels * self.in_rotations,
                *self.kernel_size,
            )
            tw = tw.view(tw_shape)

        # Apply convolution.
        input_shape = input.size()
        input = input.view(
            input_shape[0],
            self.in_channels * self.in_rotations,
            input_shape[-2],
            input_shape[-1],
        )
        input=input.float() 

        if self.hsv_space and self.in_rotations == 1:
            if self.shift_img:
                # Since we use an image stack of # out_rotations 
                # -> Repeat the kernel on the in_channel dimension with out_rotations
                # This ensures that the HSV channels all line up with their corresponding weights.
                tw = tw.repeat(1,self.out_rotations,1,1)

                # Create image stack of Hue or Sat shifted images
                # TODO FIX DAT DIT out_rot * out_rot image stack maakt ipv out_rot voor hue en sat shift (nu combineerd hij "zonder double loop")
                input = _shifted_img_stack(input, self.out_rotations, self.hue_shift, self.sat_shift)

        y = F.conv2d(
            input, weight=tw, bias=None, stride=self.stride, padding=self.padding
        )

        batch_size, _, ny_out, nx_out = y.size()
        if self.hue_shift and self.sat_shift:
            y = y.view(batch_size, self.out_channels, self.out_rotations*self.out_rotations, ny_out, nx_out)
        else:
            y = y.view(batch_size, self.out_channels, self.out_rotations, ny_out, nx_out)

        # Apply bias.
        if self.bias is not None:
            bias = self.bias.view(1, self.out_channels, 1, 1, 1)
            y = y + bias

        return y
