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
    # 计算 log_softmax（数值稳定）
    log_p_student = F.log_softmax(logits_student, dim=-1)
    log_p_teacher = F.log_softmax(logits_teacher, dim=-1)

    # 教师概率
    p_teacher = torch.exp(log_p_teacher)

    # KL(teacher || student) = Σ p_teacher * (log_p_teacher - log_p_student)
    kl = (p_teacher * (log_p_teacher - log_p_student)).sum(dim=-1)
    
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
    Top-k 反向 KL 散度估计（内存高效版，从 logits 计算）

    将概率最高的 k 个 token 精确计算，其余 token 作为一个整体（tail）计算：

        KL(teacher || student)
            ≈ Σ_{i∈top-k} p_t_i * log(p_t_i / p_s_i)
            + p_t_tail * log(p_t_tail / p_s_tail)

    其中 p_t_tail = 1 - Σ_{i∈top-k} p_t_i，p_s_tail 同理。

    与旧实现的区别：
    - 用 gather 代替全词表 mask，所有中间张量为 (B, L, k)，不再产生 (B, L, V) 的 mask、
      乘积、归一化等张量，峰值显存从 ~13 个 (B,L,V) 降至 ~2 个，接近 K3 估计。
    - 修正了旧版对 top-k 概率做再归一化的数学错误（归一化后计算的是
      KL(norm_teacher || norm_student)，不等于原始 KL 的 top-k 近似）。

    参数:
        logits_student: 学生模型的 logits, shape (B, L, V)
        logits_teacher: 教师模型的 logits, shape (B, L, V)
        k: top-k 的 k 值
        mask: 有效 token 的掩码, shape (B, L)
        epsilon: 数值稳定性常数

    返回:
        kl_per_token: KL 散度估计, shape (B, L)
    """
    # log 是单调函数，log 空间的 top-k 等价于概率空间的 top-k
    # teacher 无梯度，(B,L,V) 张量用完后可被 CUDA allocator 立即复用
    log_p_teacher = F.log_softmax(logits_teacher, dim=-1)          # (B, L, V) no grad
    top_k_log_teacher, top_k_indices = torch.topk(log_p_teacher, k=k, dim=-1)  # (B, L, k)
    del log_p_teacher  # 显式释放，让 CUDA allocator 复用此块内存

    p_teacher_top_k = top_k_log_teacher.exp()                       # (B, L, k)
    sum_p_teacher_top_k = p_teacher_top_k.sum(dim=-1)              # (B, L)
    p_teacher_tail = (1.0 - sum_p_teacher_top_k).clamp(min=epsilon)  # (B, L)

    # 学生模型：log_softmax 输出 (B,L,V) 由 autograd 保留用于反向传播（不可避免，与 K3 相同）
    log_p_student = F.log_softmax(logits_student, dim=-1)           # (B, L, V) requires_grad
    # gather 只取 k 个位置，后续所有运算均在 (B,L,k) 或 (B,L) 上进行
    top_k_log_student = torch.gather(log_p_student, dim=-1, index=top_k_indices)  # (B, L, k)

    p_student_top_k = top_k_log_student.exp()                       # (B, L, k)
    sum_p_student_top_k = p_student_top_k.sum(dim=-1)              # (B, L)
    p_student_tail = (1.0 - sum_p_student_top_k).clamp(min=epsilon)  # (B, L)

    # Top-k 部分的 KL：Σ_k p_t_k * (log p_t_k - log p_s_k)
    kl_top_k = (p_teacher_top_k * (top_k_log_teacher - top_k_log_student)).sum(dim=-1)  # (B, L)

    # Tail 部分的 KL：p_t_tail * log(p_t_tail / p_s_tail)
    kl_tail = p_teacher_tail * (p_teacher_tail / p_student_tail).clamp(min=epsilon).log()  # (B, L)

    kl = kl_top_k + kl_tail

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
    reduce: bool = True,
) -> torch.Tensor:
    """
    反向 KL 散度计算（从 logits 计算，统一接口）

    参数:
        logits_student: 学生模型的 logits, shape (B, L, V)
        logits_teacher: 教师模型的 logits, shape (B, L, V)
        config: 散度计算配置
        mask: 有效 token 的掩码, shape (B, L)
        sampled_tokens: 采样的 token IDs (用于 K3 估计), shape (B, L)
        reduce: 是否对序列维度求和。True 返回 (B,)，False 返回 (B, L)

    返回:
        kl: KL 散度, shape (B,) 或 (B, L)
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
    
    # 对序列维度求和或保留 per-token 值
    if reduce:
        return kl_per_token.sum(dim=-1)  # (B,)
    return kl_per_token  # (B, L)


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
