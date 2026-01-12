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

        # Split into first half and second half
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
    print(f"q device: {q.device}")
    print(f"rope.cos_cache device: {rope.cos_cache.device}")
    print(f"rope.sin_cache device: {rope.sin_cache.device}")
    batch, num_heads, seq_len, head_dim = q.shape
    _, num_kv_heads, _, _ = k.shape
    print(q.shape, k.shape, v.shape)
    n_groups = num_heads // num_kv_heads


    k_expanded = torch.repeat_interleave(k, n_groups, dim=1)
    v_expanded = torch.repeat_interleave(v, n_groups, dim=1)

    q_rope = rope(q)
    k_rope = rope(k_expanded) * key_weights.view(1, num_heads, 1, 1)

    print(k_rope.shape)
    print(k_rope.transpose(-2, -1).shape)
    print(q_rope.shape)
    attention = (q_rope @ k_rope.transpose(-2, -1)) * scale

    if mask is None:
        # causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))\
            .view(1, 1, seq_len, seq_len)\
            .to(device)
    attention = attention.masked_fill(mask == 0, float('-inf'))


    print("Attention scores [0,0]:")
    print(attention[0, 0])
    print("Expected pattern should be lower triangular after masking")

    a_softmaxed = F.softmax(attention, dim=-1)

    print("Softmax weights [0,0]:")
    print(a_softmaxed[0, 0])

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
        self.layer_W_k = torch.nn.Linear(hidden_dim, num_kv_heads * head_dim)
        self.layer_W_v = torch.nn.Linear(hidden_dim, num_kv_heads * head_dim)
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
        # Apply linear projections
        q = self.layer_W_q(x)
        k = self.layer_W_k(x)
        v = self.layer_W_v(x)

        # Reshape to [batch, seq_len, num_heads, head_dim] then transpose to [batch, num_heads, seq_len, head_dim]
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Calculate attention
        attention_output = calculate_attention(
            q,
            k,
            v,
            self.key_weights,
            self.rope,
            self.scale,
            x.device)

        # Reshape back to [batch, seq_len, num_heads * head_dim]
        attention_output = attention_output.transpose(1, 2).reshape(batch, seq_len, self.num_heads * self.head_dim)

        # Apply output projection
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

    print("Actual output [0,0,0]:")
    print(output[0, 0, 0])
    print("Expected output [0,0,0]:")
    print(expected[0, 0, 0])
    print("Difference:")
    print(output[0, 0, 0] - expected[0, 0, 0])
    print("Max absolute difference:")
    print(torch.max(torch.abs(output - expected)))
    diff = torch.abs(output - expected)
    wrong_positions = torch.where(diff > 1e-4)
    print("Positions with large error:")
    print(wrong_positions)
    if len(wrong_positions[0]) > 0:
        print("\nValues at first wrong position:")
        idx = (wrong_positions[0][0], wrong_positions[1][0], wrong_positions[2][0], wrong_positions[3][0])
        print(f"Actual: {output[idx]}")
        print(f"Expected: {expected[idx]}")

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

        attention_output = calculate_sliding_attention(
            q,
            k,
            v,
            self.key_weights,
            self.rope,
            self.scale,
            x.device,
            self.window_size)
        attention_output = attention_output.transpose(1, 2).reshape(batch, seq_len, self.num_heads * self.head_dim)
        ### TODO: Your code ends here ###
        return attention_output


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

        ### TODO: Your code ends here ###
        return output

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

        ### TODO: Your code ends here ###

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch, seq_len, hidden_dim].

        Returns:
            Output tensor of shape [batch, seq_len, hidden_dim].
        """
        ### TODO: Your code starts here ###

        ### TODO: Your code ends here ###

        assert x.shape == result.shape
        return result

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
