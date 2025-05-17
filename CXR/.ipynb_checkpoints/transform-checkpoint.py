import torch
import torch.nn.functional as F


def morphological_close(input_tensor, kernel_size=30):
    """
    PyTorch 实现的闭运算（先膨胀后腐蚀）
    Args:
        input_tensor: 输入二值掩码 (B, 1, H, W)，值范围为 [0, 1]
        kernel_size: 结构元素大小（椭圆半径）
    Returns:
        闭运算后的掩码
    """
    # 生成椭圆结构元素（类似OpenCV的MORPH_ELLIPSE）
    kernel = _get_ellipse_kernel(kernel_size).to(input_tensor.device)
    
    # 膨胀操作
    dilated = F.max_pool2d(input_tensor, kernel_size=kernel.shape[-1], stride=1, 
                          padding=kernel_size//2)
    
    # 腐蚀操作
    eroded = -F.max_pool2d(-dilated, kernel_size=kernel.shape[-1], stride=1, 
                          padding=kernel_size//2)
    
    return eroded[ : , : , : input_tensor.shape[2], : input_tensor.shape[3]]

def morphological_open(input_tensor, kernel_size=30):
    """
    PyTorch 实现的闭运算（先膨胀后腐蚀）
    Args:
        input_tensor: 输入二值掩码 (B, 1, H, W)，值范围为 [0, 1]
        kernel_size: 结构元素大小（椭圆半径）
    Returns:
        闭运算后的掩码
    """
    # 生成椭圆结构元素（类似OpenCV的MORPH_ELLIPSE）
    kernel = _get_ellipse_kernel(kernel_size).to(input_tensor.device)

    # 腐蚀操作
    eroded = -F.max_pool2d(-input_tensor, kernel_size=kernel.shape[-1], stride=1, 
                          padding=kernel_size//2)
    
    # 膨胀操作
    dilated = F.max_pool2d(eroded, kernel_size=kernel.shape[-1] // 2, stride=1, 
                          padding=kernel_size//4)
    
    
    return dilated[ : , : , : input_tensor.shape[2], : input_tensor.shape[3]]

def _get_ellipse_kernel(kernel_size):
    """生成椭圆结构元素（二值化的椭圆核）"""
    kernel = torch.zeros((1, 1, kernel_size, kernel_size))
    center = kernel_size // 2
    for i in range(kernel_size):
        for j in range(kernel_size):
            if (i - center)**2 + (j - center)**2 <= (center)**2:
                kernel[0, 0, i, j] = 1
    return kernel

def morphological_open_close(mask, open_kernel_size, close_kernel_size)-> torch.tensor:
    open_mask = morphological_open(mask, open_kernel_size)
    close_mask = morphological_close(open_mask, close_kernel_size)
    return close_mask
    