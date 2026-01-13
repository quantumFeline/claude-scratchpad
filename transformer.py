#[do not modify]

import torch
import torch.nn.functional as F
from typing import Optional, Callable

class SwiGLUFeedForward(torch.nn.Module):
    def __init__(self, hidden_dim: int, inner_dim: int) -> None:
        """
        Args:
            hidden_dim: Dimension of input and output tensors.
            inner_dim: Dimension of the intermediate (inner) representation.
        """
        super().__init__()

        ### TODO: Your code starts here ###
        self.layer_w1 = torch.nn.Linear(hidden_dim, inner_dim)
        self.layer_v = torch.nn.Linear(hidden_dim, inner_dim)
        self.layer_w2 = torch.nn.Linear(inner_dim, hidden_dim)
        ### TODO: Your code ends here ###

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_dim].

        Returns:
            Output tensor of shape [batch_size, seq_len, hidden_dim].
        """
        assert len(x.shape) == 3, f"Expected 3D tensor, got shape {x.shape}"

        ### TODO: Your code starts here ###
        h1 = self.layer_w1(x)
        h_v = self.layer_v(x)
        h2 = F.silu(h1) * h_v
        result = self.layer_w2(h2)
        ### TODO: Your code ends here ###

        return result

class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int = 2048, base: float = 10000.0) -> None:
        """
        Args:
            head_dim: Dimension of each attention head (must be even).
            max_seq_len: Maximum sequence length to precompute embeddings for.
            base: Base for computing rotation frequencies.

        WARNING: YOUR IMPLEMENTATION MUST PRECOMPUTE THE EMBEDDINGS
        """
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"

        ### TODO: Your code starts here ###
        n_pairs = head_dim // 2
        self.thetas = base ** (- torch.arange(n_pairs) / n_pairs)
        ### TODO: Your code ends here ###

        self._precompute_cache(max_seq_len)

    def _precompute_cache(self, seq_len: int) -> None:
        ### TODO: Your code starts here ###
        self.cos_cache = torch.cos(torch.outer(torch.arange(seq_len), self.thetas))
        self.sin_cache = torch.sin(torch.outer(torch.arange(seq_len), self.thetas))
        ### TODO: Your code ends here ###


    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch, num_heads, seq_len, head_dim].
            start_pos: Starting position index (for KV-cache during inference).

        Returns:
            Tensor with rotary embedding applied, same shape as input.
        """

        ### TODO: Your code starts here ###
        batch, num_heads, seq_len, head_dim = x.shape

        cos = self.cos_cache[start_pos:start_pos+seq_len].to(x.device)
        sin = self.sin_cache[start_pos:start_pos+seq_len].to(x.device)

        x1 = x[..., :head_dim//2]
        x2 = x[..., head_dim//2:]

        x1_rotated = x1 * cos - x2 * sin
        x2_rotated = x1 * sin + x2 * cos

        x_rotated = torch.cat([x1_rotated, x2_rotated], dim=-1)
        return x_rotated
        ### TODO: Your code ends here ###



##### TESTS START #####

@torch.no_grad()
def test_rope() -> None:
    """Test RoPE applies correct rotations."""
    head_dim = 4
    max_seq_len = 8
    batch, num_heads, seq_len = 2, 2, 4

    rope = RotaryPositionalEmbedding(head_dim, max_seq_len)
    x = torch.ones(batch, num_heads, seq_len, head_dim)

    result = rope(x)

    expected = torch.tensor(
        [[[[ 1.0000,  1.0000,  1.0000,  1.0000],
          [-0.3012,  0.9900,  1.3818,  1.0099],
          [-1.3254,  0.9798,  0.4932,  1.0198],
          [-1.1311,  0.9696, -0.8489,  1.0295]],

         [[ 1.0000,  1.0000,  1.0000,  1.0000],
          [-0.3012,  0.9900,  1.3818,  1.0099],
          [-1.3254,  0.9798,  0.4932,  1.0198],
          [-1.1311,  0.9696, -0.8489,  1.0295]]],


        [[[ 1.0000,  1.0000,  1.0000,  1.0000],
          [-0.3012,  0.9900,  1.3818,  1.0099],
          [-1.3254,  0.9798,  0.4932,  1.0198],
          [-1.1311,  0.9696, -0.8489,  1.0295]],

         [[ 1.0000,  1.0000,  1.0000,  1.0000],
          [-0.3012,  0.9900,  1.3818,  1.0099],
          [-1.3254,  0.9798,  0.4932,  1.0198],
          [-1.1311,  0.9696, -0.8489,  1.0295]]]]
    )

    assert result.shape == x.shape, f"Shape mismatch: {result.shape} vs {x.shape}"
    assert torch.allclose(result, expected, atol=1e-4), "Error in ROPE"


test_rope()

#####  TESTS END  #####

from torch.nn import attention
def calculate_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    key_weights: torch.Tensor,
    rope: RotaryPositionalEmbedding,
    scale: float,
    device: torch.device,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Args:
        q: Query tensor of shape [batch, num_heads, seq_len, head_dim].
        k: Key tensor of shape [batch, num_kv_heads, seq_len, head_dim].
        v: Value tensor of shape [batch, num_kv_heads, seq_len, head_dim].
        key_weights: Per-head key weights of shape [num_heads].
        rope: Rotary positional embedding module.
        scale: Scaling factor (typically 1/sqrt(head_dim)).
        device: Device to create the causal mask on.
        mask: Optional attention mask of shape [seq_len, seq_len]. If None, uses causal mask.

    Returns:
        Output tensor of shape [batch, num_heads, seq_len, head_dim].
    """
    ### TODO: Your code starts here ###
    batch, num_heads, seq_len, head_dim = q.shape
    _, num_kv_heads, _, _ = k.shape
    n_groups = num_heads // num_kv_heads


    k_expanded = torch.repeat_interleave(k, n_groups, dim=1)
    v_expanded = torch.repeat_interleave(v, n_groups, dim=1)

    q_rope = rope(q)
    k_rope = rope(k_expanded) * key_weights.view(1, num_heads, 1, 1)

    attention = (q_rope @ k_rope.transpose(-2, -1)) * scale

    if mask is None:
        # causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))\
            .view(1, 1, seq_len, seq_len)\
            .to(device)
    attention = attention.masked_fill(mask == 0, float('-inf'))

    a_softmaxed = F.softmax(attention, dim=-1)

    output = a_softmaxed @ v_expanded

    # output_transposed = output.transpose(1, 2) Not yet
    # output_reshaped = output_transposed.reshape(batch, seq_len, num_heads * head_dim)
    ### TODO: Your code ends here ###
    return output


class GroupedQueryAttention(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        num_kv_heads: Optional[int] = None
    ) -> None:
        """
        Args:
            hidden_dim: Input/output dimension.
            num_heads: Number of query heads.
            head_dim: Dimension of each head.
            num_kv_heads: Number of key-value heads. If None, defaults to num_heads (standard MHA).
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.scale = head_dim ** -0.5

        assert num_heads % self.num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        self.num_queries_per_kv = num_heads // self.num_kv_heads

        ### TODO: Your code starts here ###
        self.layer_W_q = torch.nn.Linear(hidden_dim, num_heads * head_dim)
        self.layer_W_k = torch.nn.Linear(hidden_dim, self.num_kv_heads * head_dim)
        self.layer_W_v = torch.nn.Linear(hidden_dim, self.num_kv_heads * head_dim)
        self.layer_W_o = torch.nn.Linear(num_heads * head_dim, hidden_dim)
        self.key_weights = torch.nn.Parameter(torch.ones(num_heads))
        self.rope = RotaryPositionalEmbedding(head_dim)
        ### TODO: Your code ends here ###

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch, seq_len, hidden_dim].

        Returns:
            Output tensor of shape [batch, seq_len, hidden_dim].
        """
        assert len(x.shape) == 3
        batch, seq_len, _ = x.shape

        ### TODO: Your code starts here ###
        q = self.layer_W_q(x)
        k = self.layer_W_k(x)
        v = self.layer_W_v(x)

        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        attention = calculate_attention(
            q,
            k,
            v,
            self.key_weights,
            self.rope,
            self.scale,
            x.device)
        attention_output = attention.transpose(1, 2).reshape(batch, seq_len, self.num_heads * self.head_dim)
        output = self.layer_W_o(attention_output)
        ### TODO: Your code ends here ###

        return output



##### TESTS START #####
@torch.no_grad()
def test_calculate_attention() -> None:
    """Test the calculate_attention function independently of module weights."""
    torch.manual_seed(42)
    batch, seq_len = 2, 4
    num_heads, num_kv_heads, head_dim = 4, 2, 4

    q = torch.randn(batch, num_heads, seq_len, head_dim)
    k = torch.randn(batch, num_kv_heads, seq_len, head_dim)
    v = torch.randn(batch, num_kv_heads, seq_len, head_dim)

    key_weights = torch.randn(num_heads)
    rope = RotaryPositionalEmbedding(head_dim)
    scale = head_dim ** -0.5

    output = calculate_attention(q, k, v, key_weights, rope, scale, q.device)

    assert output.shape == (batch, num_heads, seq_len, head_dim), \
        f"Wrong output shape: {output.shape}, expected {(batch, num_heads, seq_len, head_dim)}"

    expected = torch.tensor(
        [[[[ 1.7744, -0.9216,  0.9624, -0.3370],
          [ 0.3143, -0.2881,  0.7230,  0.4999],
          [ 0.3844,  0.4978,  0.3157,  0.0305],
          [ 0.3285,  0.6624,  0.5529,  0.1313]],

         [[ 1.7744, -0.9216,  0.9624, -0.3370],
          [-0.0153, -0.1452,  0.6690,  0.6888],
          [ 0.1277,  0.6668,  0.2440,  0.1472],
          [ 0.3011,  0.7186,  0.5328,  0.1269]],

         [[-0.8146, -1.0212, -0.4949, -0.5923],
          [-0.2359, -0.1480, -0.2879, -1.6233],
          [-0.0641,  0.5656, -0.6690, -1.0527],
          [-0.0747, -0.1315, -0.0178,  0.9354]],

         [[-0.8146, -1.0212, -0.4949, -0.5923],
          [-0.6315, -0.7449, -0.4294, -0.9186],
          [-0.6552, -0.6012, -0.6129, -0.5296],
          [ 0.1580,  0.3066, -0.0132, -1.6795]]],

        [[[-0.0045,  1.6668,  0.1539, -1.0603],
          [-0.2895,  0.8726,  0.2773,  0.4695],
          [-0.2155,  0.2792, -0.5029, -0.0560],
          [-0.1606,  0.2407, -0.3138,  0.0749]],

         [[-0.0045,  1.6668,  0.1539, -1.0603],
          [-0.2733,  0.9177,  0.2703,  0.3825],
          [-0.2363,  0.3177, -0.4039,  0.0707],
          [-0.1465,  0.1215, -0.3514,  0.1096]],

         [[-0.9291,  0.2762, -0.5389,  0.4626],
          [-0.9150,  0.2015, -0.4932,  0.7090],
          [-0.5811,  0.0459, -0.2781,  0.7316],
          [-0.2906,  0.2335, -0.0824,  0.5158]],

         [[-0.9291,  0.2762, -0.5389,  0.4626],
          [-0.8976,  0.1089, -0.4365,  1.0148],
          [-0.1522, -0.1901,  0.0176,  0.8908],
          [ 0.8761, -0.5560,  0.6310,  0.6123]]]]
    )

    assert torch.allclose(output, expected, atol=1e-4), \
        f"calculate_attention output values mismatch"


test_calculate_attention()

def calculate_sliding_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    key_weights: torch.Tensor,
    rope: RotaryPositionalEmbedding,
    scale: float,
    device: torch.device,
    window_size: int
) -> torch.Tensor:
    """
    Args:
        q: Query tensor of shape [batch, num_heads, seq_len, head_dim].
        k: Key tensor of shape [batch, num_kv_heads, seq_len, head_dim].
        v: Value tensor of shape [batch, num_kv_heads, seq_len, head_dim].
        key_weights: Per-head key weights of shape [num_heads].
        rope: Rotary positional embedding module.
        scale: Scaling factor (typically 1/sqrt(head_dim)).
        device: Device to create the causal mask on.
        window_size: Number of previous tokens each position can attend to.

    Returns:
        Output tensor of shape [batch, num_heads, seq_len, head_dim].
    """
    ### TODO: Your code starts here ###
    sliding_window_mask = torch.tril(
        torch.ones(q.shape[-2], q.shape[-2], device=device)) * \
        torch.triu(torch.ones(q.shape[-2], q.shape[-2], device=device), diagonal=-window_size)
    sliding_window_mask = sliding_window_mask.view(1, 1, q.shape[-2], q.shape[-2])

    output = calculate_attention(
        q,
        k,
        v,
        key_weights,
        rope,
        scale,
        device,
        sliding_window_mask
    )

    ### TODO: Your code ends here ###

    return output


class SWAttention(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        window_size: int
    ) -> None:
        """
        Args:
            hidden_dim: Input/output dimension.
            num_heads: Number of attention heads.
            head_dim: Dimension of each head.
            window_size: Size of the sliding window.
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.window_size = window_size
        self.scale = head_dim ** -0.5

        ### TODO: Your code starts here ###
        self.num_kv_heads = num_heads
        
        self.layer_W_q = torch.nn.Linear(hidden_dim, num_heads * head_dim)
        self.layer_W_k = torch.nn.Linear(hidden_dim, self.num_kv_heads * head_dim)
        self.layer_W_v = torch.nn.Linear(hidden_dim, self.num_kv_heads * head_dim)
        self.layer_W_o = torch.nn.Linear(num_heads * head_dim, hidden_dim)
        self.key_weights = torch.nn.Parameter(torch.ones(num_heads))
        self.rope = RotaryPositionalEmbedding(head_dim)

        ### TODO: Your code ends here ###

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch, seq_len, hidden_dim].

        Returns:
            Output tensor of shape [batch, seq_len, hidden_dim].
        """
        assert len(x.shape) == 3
        batch, seq_len, _ = x.shape

        ### TODO: Your code starts here ###
        q = self.layer_W_q(x)
        k = self.layer_W_k(x)
        v = self.layer_W_v(x)

        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        attention = calculate_sliding_attention(
            q,
            k,
            v,
            self.key_weights,
            self.rope,
            self.scale,
            x.device,
            self.window_size)
        attention_output = attention.transpose(1, 2).reshape(batch, seq_len, self.num_heads * self.head_dim)
        output = self.layer_W_o(attention_output)
        ### TODO: Your code ends here ###
        return output


##### TESTS START #####

@torch.no_grad()
def test_calculate_sliding_attention() -> None:
    """Test the calculate_sliding_attention function independently of module weights."""
    torch.manual_seed(42)
    batch, seq_len = 2, 4
    num_heads, head_dim = 4, 4
    window_size = 2

    q = torch.randn(batch, num_heads, seq_len, head_dim)
    k = torch.randn(batch, num_heads, seq_len, head_dim)
    v = torch.randn(batch, num_heads, seq_len, head_dim)

    key_weights = torch.randn(num_heads)
    rope = RotaryPositionalEmbedding(head_dim)
    scale = head_dim ** -0.5

    output = calculate_sliding_attention(q, k, v, key_weights, rope, scale, q.device, window_size)

    assert output.shape == (batch, num_heads, seq_len, head_dim), \
        f"Wrong output shape: {output.shape}, expected {(batch, num_heads, seq_len, head_dim)}"

    expected = torch.tensor(
        [[[[-6.8548e-01,  5.6356e-01, -1.5072e+00, -1.6107e+00],
          [-7.5833e-01,  5.5151e-01, -1.3803e+00, -1.3910e+00],
          [-7.5970e-01,  5.4425e-01, -1.3694e+00, -1.4018e+00],
          [-1.0289e+00, -1.5047e-03,  1.3449e-01,  8.8395e-02]],

         [[-1.3793e+00,  6.2580e-01, -2.5850e+00, -2.4000e-02],
          [-1.0804e+00,  2.9937e-01, -1.5638e+00, -4.5193e-03],
          [-3.7572e-01,  9.0874e-01, -9.8827e-01,  3.2158e-01],
          [ 5.5610e-01,  9.3138e-01,  8.2518e-01,  3.6249e-01]],

         [[ 9.7329e-01, -1.0151e+00, -5.4192e-01, -4.4102e-01],
          [ 3.7820e-01, -6.0546e-01, -6.2194e-01, -2.5908e-01],
          [ 7.5603e-01, -4.5413e-01, -2.9462e-01, -6.9975e-02],
          [ 5.6501e-01,  6.4487e-02,  4.0517e-01,  4.1787e-01]],

         [[ 4.0380e-01, -7.1398e-01,  8.3373e-01, -9.5855e-01],
          [ 4.2490e-01,  1.1594e-01, -4.9589e-01, -1.0976e+00],
          [ 3.5349e-01, -4.0529e-01, -6.6044e-01, -1.1089e+00],
          [-2.1912e-01, -6.5963e-01,  1.6555e-01, -1.0503e+00]]],


        [[[ 4.3344e-01, -7.1719e-01,  1.0554e+00, -1.4534e+00],
          [ 4.4607e-01, -2.8344e-01,  6.3300e-01, -8.4259e-01],
          [ 4.2691e-01,  3.2082e-01, -4.8548e-01, -5.2133e-01],
          [ 3.8897e-01,  6.5369e-01, -1.5100e+00, -7.1436e-01]],

         [[ 8.8538e-01,  1.8244e-01,  7.8638e-01, -5.7920e-02],
          [ 7.1542e-01, -2.9334e-01,  1.0705e-01, -3.1831e-04],
          [ 6.7078e-01,  7.1021e-01,  4.8379e-02,  5.4688e-01],
          [ 7.3988e-01,  2.3339e-01, -3.6269e-01,  3.0450e-01]],

         [[-7.9394e-01,  3.7523e-01,  8.7910e-02, -1.2415e+00],
          [-5.1264e-01, -3.4904e-01, -2.9172e-01,  6.7694e-01],
          [ 4.9596e-01,  5.8218e-01, -1.2478e-01,  1.5970e-01],
          [ 6.7310e-02,  1.5020e-01, -2.8622e-01,  1.1465e+00]],

         [[-2.1844e-01,  1.6630e-01,  2.1442e+00,  1.7046e+00],
          [ 9.8019e-02,  4.3332e-01,  8.2743e-01,  1.1330e+00],
          [ 2.0074e-02, -5.5385e-02,  2.2436e-01,  9.4053e-01],
          [-1.2419e-01,  1.2867e-01, -6.4606e-01,  2.5874e-01]]]]
    )

    assert torch.allclose(output, expected, atol=1e-4), \
        f"calculate_sliding_attention output values mismatch"

test_calculate_sliding_attention()

#####  TESTS END  #####

class Router(torch.nn.Module):
    def __init__(self, hidden_dim: int, num_experts: int, top_k: int = 2) -> None:
        """
        Args:
            hidden_dim: Input dimension.
            num_experts: Total number of experts.
            top_k: Number of experts to activate per token.
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k

        ### TODO: Your code starts here ###
        self.layer_W_g = torch.nn.Linear(hidden_dim, num_experts)
        ### TODO: Your code ends here ###

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape [batch, seq_len, hidden_dim].

        Returns:
            routing_weights: Tensor of shape [batch, seq_len, top_k] with softmax weights.
            expert_indices: Tensor of shape [batch, seq_len, top_k] with selected expert indices.
        """
        assert len(x.shape) == 3
        ### TODO: Your code starts here ###
        x_W_g = self.layer_W_g(x)
        top_k = torch.topk(x_W_g, self.top_k, dim=-1)
        g = torch.nn.functional.softmax(top_k.values, dim=-1)
        expert_indices = top_k.indices
        routing_weights = g
        ### TODO: Your code ends here ###

        return routing_weights, expert_indices


class MixtureOfExperts(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        inner_dim: int,
        num_experts: int = 8,
        top_k: int = 2
    ) -> None:
        """
        Args:
            hidden_dim: Input/output dimension.
            inner_dim: Inner dimension of each expert.
            num_experts: Total number of experts.
            top_k: Number of experts to activate per token.
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k

        ### TODO: Your code starts here ###
        self.router = Router(hidden_dim, num_experts, top_k)
        self.experts = torch.nn.ModuleList(
            [SwiGLUFeedForward(hidden_dim, inner_dim) for _ in range(num_experts)]
        )
        ### TODO: Your code ends here ###

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch, seq_len, hidden_dim].

        Returns:
            Output tensor of shape [batch, seq_len, hidden_dim].
        """
        assert len(x.shape) == 3
        batch, seq_len, hidden_dim = x.shape

        ### TODO: Your code starts here ###
        expert_indices, routing_weights = self.router(x)

        x_flat = x.view(batch * seq_len, hidden_dim)
        expert_indices_flat = expert_indices.view(batch * seq_len, self.top_k)
        routing_weights_flat = routing_weights.view(batch * seq_len, self.top_k)

        x_eval = torch.zeros_like(x_flat)
        for expert_id in range(self.num_experts):
            expert_mask = expert_indices_flat == expert_id
            token_idx, slot_idx = expert_mask.nonzero(as_tuple=True)
            
            if len(token_idx) == 0:  # No tokens for this expert
                continue

            expert_weights = routing_weights_flat[token_idx, slot_idx]
            expert_output = self.experts[expert_id](x_flat[token_idx])
            weighted_output = expert_weights.unsqueeze(-1) * expert_output
            x_eval.index_add_(0, token_idx, weighted_output)

        output = x_eval.view(batch, seq_len, hidden_dim)

        ### TODO: Your code ends here ###
        return output

from torch.nn import RMSNorm

class TransformerBlock(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        ff_dim: int,
        num_heads: int,
        head_dim: int,
        use_sliding_window: bool = False,
        window_size: int = 128,
        use_moe: bool = False,
        num_experts: int = 8,
        top_k: int = 2,
        num_kv_heads: Optional[int] = None
    ) -> None:
        """
        Args:
            hidden_dim: Hidden dimension.
            ff_dim: Feed-forward inner dimension.
            num_heads: Number of attention heads.
            head_dim: Dimension per attention head.
            use_sliding_window: Whether to use sliding window attention.
            window_size: Size of sliding window (if used).
            use_moe: Whether to use Mixture of Experts instead of single FFN.
            num_experts: Number of experts (if MoE).
            top_k: Number of experts per token (if MoE).
            num_kv_heads: Number of KV heads for GQA (None = standard MHA).
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        ### TODO: Your code starts here ###
        self.rms_norm_1 = RMSNorm(hidden_dim)
        if use_sliding_window:
            self.attention = SWAttention(hidden_dim, num_heads, head_dim, window_size)
        else:
            self.attention = GroupedQueryAttention(hidden_dim, num_heads, head_dim, num_kv_heads)
        self.rms_norm_2 = RMSNorm(hidden_dim)
        if use_moe:
            self.ffn = MixtureOfExperts(hidden_dim, ff_dim, num_experts, top_k)
        else:
            self.ffn = SwiGLUFeedForward(hidden_dim, ff_dim)
        ### TODO: Your code ends here ###

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch, seq_len, hidden_dim].

        Returns:
            Output tensor of shape [batch, seq_len, hidden_dim].
        """
        ### TODO: Your code starts here ###
        x_attention = self.attention(self.rms_norm_1(x))
        x_attention += x
        x_ffn = self.ffn(self.rms_norm_2(x_attention))
        x_ffn += x_attention
        result = x_ffn
        ### TODO: Your code ends here ###

        assert x.shape == result.shape
        return result

from torch.nn import Embedding

class Transformer(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_layers: int,
        hidden_dim: int,
        ff_dim: int,
        num_heads: int,
        head_dim: int,
        use_sliding_window_alternating: bool = False,
        window_size: int = 128,
        use_moe: bool = False,
        num_experts: int = 8,
        top_k: int = 2,
        num_kv_heads: Optional[int] = None
    ) -> None:
        """
        Args:
            vocab_size: Size of the vocabulary.
            n_layers: Number of transformer layers.
            hidden_dim: Hidden dimension.
            ff_dim: Feed-forward inner dimension.
            num_heads: Number of attention heads.
            head_dim: Dimension per attention head.
            use_sliding_window_alternating: Use sliding window on every other layer
            window_size: Size of sliding window.
            use_moe: Whether to use Mixture of Experts.
            num_experts: Number of experts (if MoE).
            top_k: Number of experts per token (if MoE).
            num_kv_heads: Number of KV heads for GQA.
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        ### TODO: Your code starts here ###
        self.embedding = Embedding(vocab_size, hidden_dim)
        self.layers = []
        use_sliding_window = False

        for i in range(n_layers):

            if use_sliding_window_alternating and i % 2 == 1:
                use_sliding_window = True
            if use_sliding_window_alternating and i % 2 == 0:
                use_sliding_window = False
              
            self.layers.append(
                TransformerBlock(
                    hidden_dim,
                    ff_dim,
                    num_heads,
                    head_dim,
                    use_sliding_window,
                    window_size,
                    use_moe,
                    num_experts,
                    top_k,
                    num_kv_heads
                    )
                )
        self.final_norm = RMSNorm(hidden_dim)
        self.output_proj = torch.nn.Linear(hidden_dim, vocab_size)
        ### TODO: Your code ends here ###

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Token indices of shape [batch, seq_len].

        Returns:
            Logits of shape [batch, seq_len, vocab_size].
        """
        assert len(x.shape) == 2, f"Expected 2D input, got shape {x.shape}"

        ### TODO: Your code starts here ###
        x = self.embedding(x)
        for block in self.layers:
            x = block(x)
        x = self.final_norm(x)
        x = self.output_proj(x)
        logits = x
        ### TODO: Your code ends here ###

        return logits


##### TESTS START #####

@torch.no_grad()
def test_transformer() -> None:
    torch.manual_seed(42)
    batch, seq_len = 2, 8
    vocab_size, n_layers, hidden_dim = 100, 2, 32
    ff_dim, num_heads, head_dim = 64, 4, 8

    model = Transformer(
        vocab_size=vocab_size,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        ff_dim=ff_dim,
        num_heads=num_heads,
        head_dim=head_dim
    )

    x = torch.randint(0, vocab_size, (batch, seq_len))
    output = model(x)

    # Test output shape
    assert output.shape == (batch, seq_len, vocab_size), f"Wrong shape: {output.shape}"

    # Test that model has correct number of layers
    assert len(model.layers) == n_layers, f"Model should have {n_layers} layers"

    # Test that model has embedding and output projection
    assert hasattr(model, 'embedding'), "Model should have embedding layer"
    assert hasattr(model, 'output_proj'), "Model should have output projection"
    assert isinstance(model.embedding, torch.nn.Embedding), "embedding should be nn.Embedding"

    # Test that model has final normalization
    assert hasattr(model, 'final_norm'), "Model should have final_norm"
    assert isinstance(model.final_norm, torch.nn.RMSNorm), "final_norm should be RMSNorm"

    # Test embedding size
    assert model.embedding.num_embeddings == vocab_size, \
        f"Embedding should have {vocab_size} tokens"
    assert model.embedding.embedding_dim == hidden_dim, \
        f"Embedding dimension should be {hidden_dim}"

    # Test output projection size
    assert model.output_proj.out_features == vocab_size, \
        f"Output projection should project to {vocab_size} dimensions"

    # Test with sliding window alternating
    model_sw = Transformer(
        vocab_size=vocab_size,
        n_layers=4,
        hidden_dim=hidden_dim,
        ff_dim=ff_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        use_sliding_window_alternating=True,
        window_size=4
    )
    output_sw = model_sw(x)
    assert output_sw.shape == (batch, seq_len, vocab_size)

    # Check that alternating layers use sliding window
    assert isinstance(model_sw.layers[1].attention, SWAttention), \
        "Layer 1 should use SWAttention (alternating pattern)"
    assert isinstance(model_sw.layers[3].attention, SWAttention), \
        "Layer 3 should use SWAttention (alternating pattern)"
    assert isinstance(model_sw.layers[0].attention, GroupedQueryAttention), \
        "Layer 0 should use GQA (not sliding window)"

    # Test with MoE
    model_moe = Transformer(
        vocab_size=vocab_size,
        n_layers=2,
        hidden_dim=hidden_dim,
        ff_dim=ff_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        use_moe=True,
        num_experts=4,
        top_k=2
    )
    output_moe = model_moe(x)
    assert output_moe.shape == (batch, seq_len, vocab_size)
    assert isinstance(model_moe.layers[0].ffn, MixtureOfExperts), \
        "All layers should use MoE when use_moe=True"

    # Test with GQA
    model_gqa = Transformer(
        vocab_size=vocab_size,
        n_layers=2,
        hidden_dim=hidden_dim,
        ff_dim=ff_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        num_kv_heads=2
    )
    output_gqa = model_gqa(x)
    assert output_gqa.shape == (batch, seq_len, vocab_size)

    # Test determinism
    torch.manual_seed(42)
    model2 = Transformer(
        vocab_size=vocab_size,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        ff_dim=ff_dim,
        num_heads=num_heads,
        head_dim=head_dim
    )
    x2 = torch.randint(0, vocab_size, (batch, seq_len))
    output2 = model2(x2)
    assert torch.allclose(output, output2, atol=1e-5), "Transformer should be deterministic"

    # Test that logits are different for different inputs
    x_different = torch.randint(0, vocab_size, (batch, seq_len))
    output_different = model(x_different)
    assert not torch.allclose(output, output_different), \
        "Different inputs should produce different outputs"


test_transformer()

#####  TESTS END  #####

#[do not modify]

import numpy as np

def generate_recursive(n_first, vocab_size, next_prob):
    assert 0 < vocab_size
    initial = np.random.randint(0, vocab_size, n_first)
    coeffs = np.random.randint(0, vocab_size, n_first)

    return initial, coeffs, vocab_size, next_prob

class SeqGen:
    """
    For generating recurrent sequences with stochastically repeating terms.
    """
    def __init__(self, initial, coeffs, size, next_prob):
        assert len(coeffs) == len(initial)
        self.initial = initial
        self.coeffs = coeffs
        self.size = size
        self.next_prob = next_prob

        self.current = initial

    def __iter__(self):
        return self

    def __next__(self):
        if np.random.random() < self.next_prob:
          new = self.current[-1] + 1
        else:
          new = (self.current @ self.coeffs)

        new %= self.size
        self.current = np.append(self.current, new)[1:]

        return new

    def __key(self):
        return (tuple(self.initial), tuple(self.coeffs), self.size, self.next_prob)

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        if isinstance(other, SeqGen):
            return self.__key() == other.__key()


def generate_dataset(gen_factory, seq_len, num_entries, exclude = []):
    """
    For generating datasets with num_entries elements each
    of length seq_len.

      gen_factory is a procedure that returns
        instance of SeqGen when called.

      seq_len is the length of the sequence to generate.

      num_entries is the number of sequences to generate.

      exclude is the set of sequences that aren't to be used in training
    """
    entries = []
    generators = []
    for e in range(num_entries):
        while True:
          seq_gen = gen_factory()
          if seq_gen in exclude:
              continue

          seq = []
          for s in range(seq_len + 1):
              seq.append(next(seq_gen))

          break

        generators.append(seq_gen)
        entries.append(seq)
    data = torch.tensor(entries, dtype=torch.long)
    x = data[:, :seq_len]
    y = data[:, 1:]       # we predict next token
    return torch.utils.data.TensorDataset(x, y), set(generators)

#[do not modify]

def example_generator(gen):
    """
      A procedure that returns a representation of
      a single data entrance.
    """
    def example_gen():
        return SeqGen(*gen())
    return example_gen

#[do not modify]

BATCH_SIZE = 128
SEQ_LEN = 64


VOCAB_SIZE = 7
NEXT_PROB = .1
INITIAL = 2


DEVICE = torch.device("cpu") # can be also cpu
PERM_EXAMPLE_GENERATOR = example_generator(lambda: generate_recursive(INITIAL, VOCAB_SIZE, NEXT_PROB))


TEST_DATASET, generators = generate_dataset(
    gen_factory=PERM_EXAMPLE_GENERATOR, seq_len=SEQ_LEN, num_entries=1000)
TRAIN_DATASET, _ = generate_dataset(
    gen_factory=PERM_EXAMPLE_GENERATOR, seq_len=SEQ_LEN, num_entries=10000, exclude=generators)


TRAIN_LOADER = torch.utils.data.DataLoader(
    TRAIN_DATASET, batch_size=BATCH_SIZE)
TEST_LOADER = torch.utils.data.DataLoader(TEST_DATASET, batch_size=BATCH_SIZE)

from tqdm import tqdm
import functools


@torch.no_grad()
def eval_acc(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader):
    model.eval()
    sum_acc = 0
    num_examples = 0
    for x, y in dataloader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        model_out = model(x)

        acc = (torch.argmax(model_out, dim=-1) == y).to(torch.float32).sum()
        sum_acc += acc
        num_examples += model_out.shape[0] * model_out.shape[1]

    return sum_acc / num_examples


def eval_fn(step, model, dataloader):
    acc = eval_acc(model, dataloader)
    print(f"{step}: Avg eval accuracy {acc}")
    return acc


def train(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        dataloader: torch.utils.data.DataLoader,
        eval_fn: functools.partial,
        num_epochs: int
):
    model.train()

    train_losses = []
    eval_accuracies = []

    for epoch in range(num_epochs):
        if epoch == 0:
            current_eval_acc = eval_fn(epoch, model)
            eval_accuracies.append(current_eval_acc.item())

        model.train()
        total_loss = 0.0
        num_batches = 0
        for i, (x, y) in tqdm(enumerate(dataloader)):
            ### TODO: Your code starts here ###
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            output = model(x)
            loss = torch.nn.functional.cross_entropy(output.reshape(-1, output.shape[-1]), y.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
            ### TODO: Your code ends here ###

        avg_train_loss = total_loss / num_batches
        train_losses.append(avg_train_loss)

        current_eval_acc = eval_fn(epoch, model)
        eval_accuracies.append(current_eval_acc.item())

    return train_losses, eval_accuracies


model = Transformer(
    vocab_size=VOCAB_SIZE, n_layers=4, hidden_dim=64, ff_dim=128, num_heads=4, head_dim=16
)
model.to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

train_losses_history, eval_accuracies_history = train(
    model=model,
    optimizer=optimizer,
    dataloader=TRAIN_LOADER,
    eval_fn=functools.partial(
        eval_fn,
        dataloader=TEST_LOADER,
    ),
    num_epochs=8
)

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses_history, label='Training Loss', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss per Epoch')
plt.legend()
plt.grid(True)

# Plotting the evaluation accuracy
plt.subplot(1, 2, 2)
plt.plot(eval_accuracies_history, label='Evaluation Accuracy', color='blue')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Evaluation Accuracy per Epoch')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

@torch.no_grad()
def token_choice_greedy(model_logits: torch.Tensor) -> torch.Tensor:
    """
    Select the most likely token (greedy decoding).

    Args:
        model_logits: Logits of shape [batch, seq_len, vocab_size].

    Returns:
        Selected token indices of shape [batch, 1].
    """
    assert len(model_logits.shape) == 3
    return torch.argmax(model_logits[:, -1:, :], dim=-1)


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    input: torch.Tensor,
    gen_length: int,
    token_choice: Callable[[torch.Tensor], torch.Tensor] = token_choice_greedy
) -> torch.Tensor:
    """
    Generate new tokens autoregressively.

    Args:
        model: The transformer model.
        input: Initial token sequence of shape [batch, seq_len].
        gen_length: Number of tokens to generate.
        token_choice: Function to select next token from logits.

    Returns:
        Generated tokens of shape [batch, gen_length] (without the input).
    """
    assert len(input.shape) == 2
    model.eval()

    current_seq = input.to(DEVICE)
    output_tokens = []

    for _ in range(gen_length):
        ### TODO: Your code starts here ###
        model_out = model(current_seq)
        next_token = token_choice(model_out)
        output_tokens.append(next_token)
        current_seq = torch.cat((current_seq, next_token), dim=-1)
        ### TODO: Your code ends here ###

    return torch.cat(output_tokens, dim=-1)


print(generate(model, torch.tensor([[1]]), 100))


@torch.no_grad()
def get_dist_after_with_temp_and_topp(
        model_logits: torch.Tensor, top_p: float, t: float
) -> torch.Tensor:
    assert len(model_logits.shape) == 3

    ### TODO: Your code starts here ###
    probs = torch.nn.functional.softmax(model_logits / t, dim=-1)
    sorted_probs, prob_indices = torch.sort(probs, descending=True, dim=-1)
    probs_cumulative = torch.cumsum(sorted_probs, dim=-1)
    mask = probs_cumulative < top_p
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = True
    probs_top_p = sorted_probs * mask

    temp_result = torch.zeros_like(probs)
    temp_result.scatter_(dim=-1, index=prob_indices, src=probs_top_p)

    result = temp_result / (torch.sum(temp_result, dim=-1, keepdim=True) + 1e-6)
    return result
    ### TODO: Your code ends here ###


@torch.no_grad()
def token_choice_adv(
        model_logits: torch.Tensor, top_p: float, t: float
) -> torch.Tensor:
    probs = get_dist_after_with_temp_and_topp(
        model_logits=model_logits[:, -1:, :], top_p=top_p, t=t
    )
    dist = torch.distributions.Categorical(probs=probs)
    return dist.sample()


##### TESTS START #####
@torch.no_grad()
def test_nucleus() -> None:
    def test_dist_topp() -> None:
        res = get_dist_after_with_temp_and_topp(
            torch.tensor([[[1.0, 1.0, 1.0]]]), top_p=1.0, t=1.0
        )
        assert torch.abs(res - torch.tensor([1 / 3, 1 / 3, 1 / 3])).sum() <= 1e-4

        res = get_dist_after_with_temp_and_topp(
            torch.tensor([[[2.0, 3.0, 1.0]]]), top_p=0.0, t=1.0
        )

        assert torch.abs(res - torch.tensor([0.0, 1.0, 0.0])).sum() <= 1e-4

        res = get_dist_after_with_temp_and_topp(
            torch.tensor([[[2.0, 3.0, 1.0]]]), top_p=0.6, t=1.0
        )
        assert torch.abs(res - torch.tensor([0.0, 1.0, 0.0])).sum() <= 1e-4

        res = get_dist_after_with_temp_and_topp(
            torch.tensor([[[1.0, 3.0, 2.0]]]), top_p=0.71, t=1.0
        )
        assert torch.abs(res - torch.tensor([0.0, 0.7311, 0.2689])).sum() <= 1e-2

        res = get_dist_after_with_temp_and_topp(
            torch.tensor([[[1.0, 3.0, 2.0]]]), top_p=1.0, t=1.0
        )
        assert torch.abs(res - torch.tensor([0.0900, 0.6652, 0.2447])).sum() <= 1e-2

    def test_temperature() -> None:
        res = get_dist_after_with_temp_and_topp(
            torch.tensor([[[1.0, 1.0, 1.0]]]), top_p=1.0, t=3.0
        )
        assert torch.abs(res - torch.tensor([1 / 3, 1 / 3, 1 / 3])).sum() <= 1e-4

        res = get_dist_after_with_temp_and_topp(
            torch.tensor([[[1.0, 3.0, 2.0]]]), top_p=1.0, t=3.0
        )
        assert torch.abs(res - torch.tensor([0.2302, 0.4484, 0.3213])).sum() <= 1e-2

        res = get_dist_after_with_temp_and_topp(
            torch.tensor([[[1.0, 3.0, 2.0]]]), top_p=1.0, t=1 / 3
        )
        assert torch.abs(res - torch.tensor([0.0024, 0.9503, 0.0473])).sum() <= 1e-2

        res = get_dist_after_with_temp_and_topp(
            torch.tensor([[[1.0, 3.0, 2.0]]]), top_p=0.94, t=1 / 3
        )
        assert torch.abs(res - torch.tensor([0.0, 1.0, 0.0])).sum() <= 1e-4

    def test_batching() -> None:
        res = get_dist_after_with_temp_and_topp(
            torch.tensor([
                [[1.0, 3.0, 2.0], [1.0, 4.0, 8.0]],
                [[8.0, 4.0, 1.0], [3.0, 1.0, 2.0]]
            ]),
            top_p=0.7,
            t=1.0,
        )
        expected = torch.tensor([
            [[0.0, 0.7311, 0.2689], [0.0, 0.0, 1.0]],
            [[1.0, 0.0, 0.0], [0.7311, 0.0, 0.2689]],
        ])
        assert torch.abs(res - expected).sum() <= 1e-2

    test_dist_topp()
    test_temperature()
    test_batching()


test_nucleus()

#####  TESTS END  #####

input_data = torch.ones((1, SEQ_LEN-10), dtype=int) * 2
print(generate(model=model, input=input_data, gen_length=SEQ_LEN - input_data.shape[-1], token_choice=functools.partial(token_choice_adv, top_p=0.5, t=0.7)))

def sequence_entropy(sequences):
    """Calculate entropy for each sequence."""
    entropies = []
    for seq in sequences:
        counts = torch.bincount(seq, minlength=8)[1:]  # classes 1-7, skip 0
        probs = counts.float() / counts.sum()
        probs = probs[probs > 0]
        entropy = -(probs * torch.log(probs)).sum()
        entropies.append(entropy)
    return torch.tensor(entropies)

entropies = []
temps = torch.linspace(1e-6, 10, 20)

num_generations = 100
batched_input_data = torch.ones((num_generations, SEQ_LEN - input_data.shape[-1]), dtype=int) * 2

for t in tqdm(temps):
    generated_sequences = generate(
        model=model,
        input=batched_input_data,
        gen_length=SEQ_LEN - batched_input_data.shape[-1],
        token_choice=functools.partial(token_choice_adv, top_p=0.5, t=t)
    )
    seq_entropies = sequence_entropy(generated_sequences)
    mean_entropy = seq_entropies.mean()
    entropies.append(mean_entropy)

plt.plot(temps, entropies)
plt.xlabel("Temperature")
plt.ylabel("Entropy")
plt.title("Entropy of Generated Sequences")

import matplotlib.pyplot as plt


@torch.no_grad()
def calc_per_token_acc(
        model: torch.nn.Module, data_loader: torch.utils.data.DataLoader
) -> np.ndarray:
    """
    Calculate accuracy for each position in the sequence.

    Args:
        model: The transformer model.
        data_loader: Data loader with (x, y) pairs.

    Returns:
        Array of shape [seq_len] with accuracy at each position.
    """
    model.eval()

    ### TODO: Your code starts here ###
    seq_len = SEQ_LEN

    per_token_correct = torch.zeros((seq_len,))
    per_token_total = 0

    for x, y in tqdm(data_loader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        output = model(x)

        per_token_correct += (torch.argmax(output, dim=-1) == y).to(torch.float32).sum(dim=0)
        per_token_total += x.shape[0]

    per_token_acc = per_token_correct / per_token_total

    ### TODO: Your code ends here ###
    return per_token_acc


per_token_acc = calc_per_token_acc(model, TEST_LOADER)
plt.figure(figsize=(10, 5))
plt.plot(np.arange(per_token_acc.shape[0]), per_token_acc)
plt.xlabel("Position in sequence")
plt.ylabel("Accuracy")
plt.title("Per-Position Prediction Accuracy")
plt.grid(True, alpha=0.3)
plt.show()
