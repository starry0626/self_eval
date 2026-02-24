import torch
import torch.nn.functional as F
from typing import Optional, Literal
from dataclasses import dataclass


@dataclass
class DivergenceConfig:
    """散度计算配置"""
    method: Literal["full", "top_k", "k3"] = "full"
    top_k: int = 20
    epsilon: float = 1e-8


def compute_full_reverse_kl_from_logits(
    logits_student: torch.Tensor,
    logits_teacher: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    epsilon: float = 1e-8
) -> torch.Tensor:
    """
    完整反向 KL 散度计算（从 logits 计算）
    
    KL(teacher || student) = Σ teacher(x) * log(teacher(x) / student(x))
                           = Σ teacher(x) * (log_teacher(x) - log_student(x))
    
    参数:
        logits_student: 学生模型的 logits, shape (B, L, V)
        logits_teacher: 教师模型的 logits, shape (B, L, V)
        mask: 有效 token 的掩码, shape (B, L)
        epsilon: 数值稳定性常数
    
    返回:
        kl_per_token: KL 散度, shape (B, L)
    """
    # 计算 log_softmax
    log_p_student = F.log_softmax(logits_student, dim=-1)
    log_p_teacher = F.log_softmax(logits_teacher, dim=-1)
    
    # 转换为概率
    p_student = torch.exp(log_p_student)
    p_teacher = torch.exp(log_p_teacher)
    
    # 添加数值稳定性
    p_student = p_student + epsilon
    p_teacher = p_teacher + epsilon
    
    # 归一化
    p_student = p_student / p_student.sum(dim=-1, keepdim=True)
    p_teacher = p_teacher / p_teacher.sum(dim=-1, keepdim=True)
    
    # 计算 log 比值
    log_ratio = log_p_teacher - log_p_student
    
    # KL = Σ p_teacher * log(p_teacher / p_student)
    kl = p_teacher * log_ratio
    
    # 在词表维度求和
    kl = kl.sum(dim=-1)
    
    # 应用 mask
    if mask is not None:
        kl = kl * mask
    
    return kl


def compute_top_k_reverse_kl_from_logits(
    logits_student: torch.Tensor,
    logits_teacher: torch.Tensor,
    k: int = 20,
    mask: Optional[torch.Tensor] = None,
    epsilon: float = 1e-8
) -> torch.Tensor:
    """
    Top-k 反向 KL 散度估计（从 logits 计算）
    
    将概率最高的 k 个 token 单独计算，其余 token 作为一个整体（tail）计算。
    
    公式:
    L ≈ Σ_top_k π(y_t|x,y<t) * log(π(y_t|x,y<t) / stopgrad(q(y_t|x,f,y<t)))
      + tail * log(tail / stopgrad(1 - Σ_top_k q(y_t|x,f,y<t)))
    
    参数:
        logits_student: 学生模型的 logits, shape (B, L, V)
        logits_teacher: 教师模型的 logits, shape (B, L, V)
        k: top-k 的 k 值
        mask: 有效 token 的掩码, shape (B, L)
        epsilon: 数值稳定性常数
    
    返回:
        kl_per_token: KL 散度估计, shape (B, L)
    """
    # 计算概率
    p_student = torch.softmax(logits_student, dim=-1)  # (B, L, V)
    p_teacher = torch.softmax(logits_teacher, dim=-1)  # (B, L, V)
    
    # 找到 top-k 的索引
    _, top_k_indices = torch.topk(p_teacher, k=k, dim=-1)  # (B, L, k)
    
    # 创建 mask
    top_k_mask = torch.zeros_like(p_teacher)
    top_k_mask.scatter_(-1, top_k_indices, 1.0)
    
    # Top-k 部分
    p_student_top_k = p_student * top_k_mask
    p_teacher_top_k = p_teacher * top_k_mask
    
    # Tail 部分 (1 - sum(top_k))
    p_student_tail = p_student * (1 - top_k_mask)
    p_teacher_tail = p_teacher * (1 - top_k_mask)
    
    # Top-k 部分的 sum
    sum_p_student_top_k = p_student_top_k.sum(dim=-1)  # (B, L)
    sum_p_teacher_top_k = p_teacher_top_k.sum(dim=-1)  # (B, L)
    
    # Tail 部分的 sum
    sum_p_student_tail = p_student_tail.sum(dim=-1)  # (B, L)
    sum_p_teacher_tail = p_teacher_tail.sum(dim=-1)  # (B, L)
    
    # 添加 epsilon
    sum_p_student_top_k = sum_p_student_top_k + epsilon
    sum_p_teacher_top_k = sum_p_teacher_top_k + epsilon
    sum_p_student_tail = sum_p_student_tail + epsilon
    sum_p_teacher_tail = sum_p_teacher_tail + epsilon
    
    # 归一化
    p_student_top_k_norm = p_student_top_k / sum_p_student_top_k.unsqueeze(-1)
    p_teacher_top_k_norm = p_teacher_top_k / sum_p_teacher_top_k.unsqueeze(-1)
    
    # 计算 top-k 部分的 KL
    # KL_top_k = Σ top_k p_teacher * log(p_teacher / p_student)
    log_ratio_top_k = p_teacher_top_k_norm.clamp(min=epsilon).log() - p_student_top_k_norm.clamp(min=epsilon).log()
    kl_top_k = p_teacher_top_k_norm * log_ratio_top_k
    kl_top_k = kl_top_k.sum(dim=-1)
    
    # 计算 tail 部分的 KL
    # KL_tail = p_teacher_tail * log(p_teacher_tail / p_student_tail)
    log_ratio_tail = (sum_p_teacher_tail / sum_p_student_tail).clamp(min=epsilon).log()
    kl_tail = sum_p_teacher_tail * log_ratio_tail
    
    # 合并
    kl = kl_top_k + kl_tail
    
    # 应用 mask
    if mask is not None:
        kl = kl * mask
    
    return kl


def compute_k3_reverse_kl_from_sampled_tokens(
    logits_student: torch.Tensor,
    logits_teacher: torch.Tensor,
    sampled_tokens: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    epsilon: float = 1e-8
) -> torch.Tensor:
    """
    K3 反向 KL 散度估计（从 logits 计算）
    
    只考虑采样到的 token 的估计。
    
    公式: k_3(x) = q(x)/pθ(x) - 1 - log(q(x)/pθ(x))
    
    参数:
        logits_student: 学生模型的 logits, shape (B, L, V)
        logits_teacher: 教师模型的 logits, shape (B, L, V)
        sampled_tokens: 采样的 token IDs, shape (B, L)
        mask: 有效 token 的掩码, shape (B, L)
        epsilon: 数值稳定性常数
    
    返回:
        kl_per_token: KL 散度估计, shape (B, L)
    """
    # 计算 log_softmax
    log_p_student = F.log_softmax(logits_student, dim=-1)  # (B, L, V)
    log_p_teacher = F.log_softmax(logits_teacher, dim=-1)  # (B, L, V)
    
    # 获取采样位置的概率
    log_p_student_sampled = torch.gather(
        log_p_student,
        dim=-1,
        index=sampled_tokens.unsqueeze(-1)
    ).squeeze(-1)  # (B, L)
    
    log_p_teacher_sampled = torch.gather(
        log_p_teacher,
        dim=-1,
        index=sampled_tokens.unsqueeze(-1)
    ).squeeze(-1)  # (B, L)
    
    # 转换为概率
    p_student_sampled = torch.exp(log_p_student_sampled).clamp(min=epsilon)
    p_teacher_sampled = torch.exp(log_p_teacher_sampled).clamp(min=epsilon)
    
    # 计算 k3 公式: q/p - 1 - log(q/p)
    # = q/p - 1 - (log q - log p)
    ratio = p_teacher_sampled / p_student_sampled
    k3 = ratio - 1 - (log_p_teacher_sampled - log_p_student_sampled)
    
    # 应用 mask
    if mask is not None:
        k3 = k3 * mask
    
    return k3


def compute_reverse_kl(
    logits_student: torch.Tensor,
    logits_teacher: torch.Tensor,
    config: Optional[DivergenceConfig] = None,
    mask: Optional[torch.Tensor] = None,
    sampled_tokens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    反向 KL 散度计算（从 logits 计算，统一接口）
    
    参数:
        logits_student: 学生模型的 logits, shape (B, L, V)
        logits_teacher: 教师模型的 logits, shape (B, L, V)
        config: 散度计算配置
        mask: 有效 token 的掩码, shape (B, L)
        sampled_tokens: 采样的 token IDs (用于 K3 估计), shape (B, L)
    
    返回:
        kl: KL 散度, shape (B,)
    """
    config = config or DivergenceConfig()
    epsilon = config.epsilon
    
    if config.method == "full":
        kl_per_token = compute_full_reverse_kl_from_logits(
            logits_student, logits_teacher, mask, epsilon
        )
    elif config.method == "top_k":
        kl_per_token = compute_top_k_reverse_kl_from_logits(
            logits_student, logits_teacher, config.top_k, mask, epsilon
        )
    elif config.method == "k3":
        if sampled_tokens is None:
            raise ValueError("K3 estimation requires sampled_tokens")
        kl_per_token = compute_k3_reverse_kl_from_sampled_tokens(
            logits_student, logits_teacher, sampled_tokens, mask, epsilon
        )
    else:
        raise ValueError(f"Unknown method: {config.method}")
    
    # 对序列维度求和，返回 (B,)
    return kl_per_token.sum(dim=-1)


class ReverseKLLoss(torch.nn.Module):
    """反向 KL 散度损失模块"""
    
    def __init__(
        self,
        method: Literal["full", "top_k", "k3"] = "full",
        top_k: int = 20,
        epsilon: float = 1e-8
    ):
        super().__init__()
        self.config = DivergenceConfig(
            method=method,
            top_k=top_k,
            epsilon=epsilon
        )
    
    def forward(
        self,
        logits_student: torch.Tensor,
        logits_teacher: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        计算反向 KL 散度损失
        
        参数:
            logits_student: 学生模型的 logits, shape (B, L, V)
            logits_teacher: 教师模型的 logits, shape (B, L, V)
            labels: token IDs, shape (B, L)
            mask: 有效 token 的掩码, shape (B, L)
        
        返回:
            loss: 标量损失
        """
        # 计算 KL 散度
        kl = compute_reverse_kl(
            logits_student=logits_student,
            logits_teacher=logits_teacher,
            config=self.config,
            mask=mask,
            sampled_tokens=labels
        )
        
        # 返回均值
        return kl.mean()
