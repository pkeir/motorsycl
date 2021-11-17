#ifndef _PGI_CUDA_HPP_
#define _PGI_CUDA_HPP_

extern "C" {

extern int __pgi_gangidx(void);
extern int __pgi_workeridx(void);
extern int __pgi_vectoridx(void);
extern int __pgi_blockidx(int);
extern int __pgi_threadidx(int);

// Found in binary files such as libnvc.a, but the last two are
// also in nvhpc_cuda_runtime.h, though including that produces errors
extern int __pgi_numgangs(void);
extern int __pgi_numworkers(void);
extern int __pgi_vectorlength(void);
extern int __pgi_numblocks(int);
extern int __pgi_numthreads(int);

}

#define _THREADIDX_X __pgi_threadidx(1)
#define _THREADIDX_Y __pgi_threadidx(2)
#define _THREADIDX_Z __pgi_threadidx(3)

#define _BLOCKIDX_X  __pgi_blockidx(1)
#define _BLOCKIDX_Y  __pgi_blockidx(2)
#define _BLOCKIDX_Z  __pgi_blockidx(3)

#define _BLOCKDIM_X  __pgi_numthreads(1)
#define _BLOCKDIM_Y  __pgi_numthreads(2)
#define _BLOCKDIM_Z  __pgi_numthreads(3)

#define _GRIDDIM_X   __pgi_numblocks(1)
#define _GRIDDIM_Y   __pgi_numblocks(2)
#define _GRIDDIM_Z   __pgi_numblocks(3)

#if 0
#define _THREADIDX_X threadIdx.x
#define _THREADIDX_Y threadIdx.y
#define _THREADIDX_Z threadIdx.z

#define _BLOCKIDX_X  blockIdx.x
#define _BLOCKIDX_Y  blockIdx.y
#define _BLOCKIDX_Z  blockIdx.z

#define _BLOCKDIM_X  blockDim.x
#define _BLOCKDIM_Y  blockDim.y
#define _BLOCKDIM_Z  blockDim.z

#define _GRIDDIM_X   gridDim.x
#define _GRIDDIM_Y   gridDim.y
#define _GRIDDIM_Z   gridDim.z
#endif

#endif // _PGI_CUDA_HPP_
