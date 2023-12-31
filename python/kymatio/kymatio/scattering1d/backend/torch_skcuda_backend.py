import torch
import cupy
from collections import namedtuple
from string import Template

from ...backend.torch_skcuda_backend import TorchSkcudaBackend
from .torch_backend import TorchBackend1D


# As of v8, cupy.util has been renamed cupy._util.
if hasattr(cupy, '_util'):
    memoize = cupy._util.memoize
else:
    memoize = cupy.util.memoize

@memoize(for_each_device=True)
def load_kernel(kernel_name, code, **kwargs):
    code = Template(code).substitute(**kwargs)
    kernel_code = cupy.cuda.compile_with_cache(code)
    return kernel_code.get_function(kernel_name)

Stream = namedtuple('Stream', ['ptr'])

def get_dtype(t):
    if isinstance(t, torch.cuda.FloatTensor):
        return 'float'
    elif isinstance(t, torch.cuda.DoubleTensor):
        return 'double'


class SubsampleFourier(object):
    """Subsampling in the Fourier domain

    Subsampling in the temporal domain amounts to periodization in the Fourier
    domain, so the input is periodized according to the subsampling factor.

    Usage
    -----
    sub_fourier = SubsampleFourier()
    res = sub_fourier(x, 8)

    Parameters
    ----------
    x : tensor
        Input tensor with at least 3 dimensions, where the next to last
        corresponds to the frequency index in the standard PyTorch FFT
        ordering. The length of this dimension should be a power of 2 to
        avoid errors. The last dimension should represent the real and
        imaginary parts of the Fourier transform.
    k : int
        The subsampling factor.

    Returns
    -------
    res : tensor
        The input tensor periodized along the next to last axis to yield a
        tensor of size x.shape[-2] // k along that dimension.
    """
    def __init__(self, backend='skcuda'):
        self.block = (1024, 1, 1)
        self.backend = backend

    def get_blocks(self, N, threads):
        return (N + threads - 1) // threads

    def __call__(self, x, k):
        if not x.is_cuda and self.backend == 'skcuda':
            raise TypeError('Use the torch backend (without skcuda) for CPU tensors.')


        out = torch.empty(x.shape[:-2] + (x.shape[-2] // k, x.shape[-1]), dtype=x.dtype, layout=x.layout, device=x.device)

        kernel = '''
        #define NT ${T} / ${k}
        extern "C"
        __global__ void periodize(const ${dtype}2 *input, ${dtype}2 *output)
        {
          int tx = blockIdx.x * blockDim.x + threadIdx.x;
          int ty = blockIdx.y * blockDim.y + threadIdx.y;

          if(tx >= NT || ty >= ${B})
            return;
          input += ty * ${T} + tx;
          ${dtype}2 res = make_${dtype}2(0.f, 0.f);

            for (int i=0; i<${k}; ++i)
            {
              const ${dtype}2 &c = input[i * NT];
              res.x += c.x;
              res.y += c.y;
            }
          res.x /= ${k};
          res.y /= ${k};
          output[ty * NT + tx] = res;
        }
        '''
        B = x.shape[0] * x.shape[1]
        T = x.shape[2]
        periodize = load_kernel('periodize', kernel, B=B, T=T, k=k, dtype=get_dtype(x))
        grid = (self.get_blocks(out.shape[-2], self.block[0]),
                self.get_blocks(out.nelement() // (2*out.shape[-2]), self.block[1]),
                1)
        periodize(grid=grid, block=self.block, args=[x.data_ptr(), out.data_ptr()],
                  stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))

        return out


class TorchSkcudaBackend1D(TorchSkcudaBackend, TorchBackend1D):
    _subsample_fourier = SubsampleFourier()

    @classmethod
    def modulus(cls, x):
        """Compute the complex modulus

            Computes the modulus of x and stores the result in a complex tensor of the
            same size, with the real part equal to the modulus and the imaginary part
            equal to zero.

            Parameters
            ----------
            x : tensor
                A complex tensor (that is, whose last dimension is equal to 2).

            Returns
            -------
            norm : tensor
                A tensor with the same dimensions as x, such that norm[..., 0] contains
                the complex modulus of x, while norm[..., 1] = 0.
        """
        cls.complex_check(x)
        return torch.abs(x)

    @classmethod
    def subsample_fourier(cls, x, k):
        """Subsampling in the Fourier domain

        Subsampling in the temporal domain amounts to periodization in the Fourier
        domain, so the input is periodized according to the subsampling factor.

        Parameters
        ----------
        x : tensor
            Input tensor with at least 3 dimensions, where the next to last
            corresponds to the frequency index in the standard PyTorch FFT
            ordering. The length of this dimension should be a power of 2 to
            avoid errors. The last dimension should represent the real and
            imaginary parts of the Fourier transform.
        k : int
            The subsampling factor.

        Returns
        -------
        res : tensor
            The input tensor periodized along the next to last axis to yield a
            tensor of size x.shape[-2] // k along that dimension.
        """
        cls.complex_check(x)
        return cls._subsample_fourier(x,k)


backend = TorchSkcudaBackend1D
