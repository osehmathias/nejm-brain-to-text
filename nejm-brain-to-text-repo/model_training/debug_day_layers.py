"""
Debug script to verify day layer implementations are equivalent
between ParameterList (RNN) and stacked tensor (Conformer) approaches.
"""
import torch
import torch.nn as nn

print("=" * 60)
print("TEST 1: Day Layer Tensor Indexing Equivalence")
print("=" * 60)

n_days = 45
neural_dim = 512
batch_size = 32

# Approach 1: ParameterList (RNN style)
day_weights_list = nn.ParameterList(
    [nn.Parameter(torch.randn(neural_dim, neural_dim)) for _ in range(n_days)]
)
day_biases_list = nn.ParameterList(
    [nn.Parameter(torch.randn(1, neural_dim)) for _ in range(n_days)]
)

# Approach 2: Stacked tensor (Conformer style) - copy same values
day_weights_tensor = nn.Parameter(
    torch.stack([day_weights_list[i].data.clone() for i in range(n_days)], dim=0)
)
day_biases_tensor = nn.Parameter(
    torch.stack([day_biases_list[i].data.clone() for i in range(n_days)], dim=0)
)

# Create day indices
day_idx = torch.tensor([0, 5, 10, 15, 20, 25, 30, 35, 40, 44] * 3 + [0, 1])

print(f"day_idx dtype: {day_idx.dtype}")
print(f"day_idx range: [{day_idx.min()}, {day_idx.max()}]")

# Approach 1: Loop iteration
weights_loop = torch.stack([day_weights_list[i] for i in day_idx], dim=0)
biases_loop = torch.cat([day_biases_list[i] for i in day_idx], dim=0).unsqueeze(1)

# Approach 2: Tensor indexing
weights_tensor = day_weights_tensor[day_idx]
biases_tensor = day_biases_tensor[day_idx]

print(f"\nShapes - weights_loop: {weights_loop.shape}, weights_tensor: {weights_tensor.shape}")
print(f"Shapes - biases_loop: {biases_loop.shape}, biases_tensor: {biases_tensor.shape}")

weights_match = torch.allclose(weights_loop, weights_tensor)
biases_match = torch.allclose(biases_loop, biases_tensor)

print(f"\n[TEST 1a] Weights match: {weights_match}")
print(f"[TEST 1b] Biases match: {biases_match}")

if not weights_match:
    print(f"  ERROR - Max weight diff: {(weights_loop - weights_tensor).abs().max()}")
if not biases_match:
    print(f"  ERROR - Max bias diff: {(biases_loop - biases_tensor).abs().max()}")

print("\n" + "=" * 60)
print("TEST 2: Gradient Flow Comparison")
print("=" * 60)

seq_len = 200
torch.manual_seed(42)

# Fresh params
day_weights_list2 = nn.ParameterList(
    [nn.Parameter(torch.eye(neural_dim) + 0.01 * torch.randn(neural_dim, neural_dim)) for _ in range(n_days)]
)
day_weights_tensor2 = nn.Parameter(
    torch.stack([day_weights_list2[i].data.clone() for i in range(n_days)], dim=0)
)

x = torch.randn(batch_size, seq_len, neural_dim)
day_idx = torch.randint(0, n_days, (batch_size,))

# Forward + backward - ParameterList style
x1 = x.clone().requires_grad_(True)
weights1 = torch.stack([day_weights_list2[i] for i in day_idx], dim=0)
out1 = torch.einsum("btd,bdk->btk", x1, weights1)
loss1 = out1.sum()
loss1.backward()

list_grads = []
list_grad_norms = []
for i in range(n_days):
    if day_weights_list2[i].grad is not None:
        list_grads.append(day_weights_list2[i].grad.clone())
        list_grad_norms.append(day_weights_list2[i].grad.norm().item())
    else:
        list_grads.append(torch.zeros_like(day_weights_list2[i]))
        list_grad_norms.append(0.0)

# Forward + backward - Tensor style
x2 = x.clone().requires_grad_(True)
weights2 = day_weights_tensor2[day_idx]
out2 = torch.einsum("btd,bdk->btk", x2, weights2)
loss2 = out2.sum()
loss2.backward()

tensor_grad = day_weights_tensor2.grad

print(f"Unique days in batch: {sorted(day_idx.unique().tolist())}")
print(f"Number of unique days: {len(day_idx.unique())}")

# Compare
list_grad_stack = torch.stack(list_grads, dim=0)
grads_match = torch.allclose(list_grad_stack, tensor_grad, atol=1e-6)

print(f"\n[TEST 2a] Gradients match: {grads_match}")

if not grads_match:
    diff = (list_grad_stack - tensor_grad).abs()
    print(f"  Max gradient difference: {diff.max():.2e}")

    # Check per-day
    for i in range(n_days):
        day_diff = (list_grads[i] - tensor_grad[i]).abs().max()
        if day_diff > 1e-6:
            print(f"  Day {i}: diff = {day_diff:.2e}")

# Gradient statistics
days_with_grad_list = sum(1 for n in list_grad_norms if n > 0)
days_with_grad_tensor = (tensor_grad.abs().sum(dim=[1,2]) > 0).sum().item()

print(f"\n[TEST 2b] Days with gradients (ParameterList): {days_with_grad_list}/{n_days}")
print(f"[TEST 2c] Days with gradients (Tensor): {days_with_grad_tensor}/{n_days}")

print(f"\nGradient norm (Tensor total): {tensor_grad.norm():.4f}")
print(f"Gradient norm (List total): {list_grad_stack.norm():.4f}")

print("\n" + "=" * 60)
print("TEST 3: Full Forward Pass Equivalence (with activation)")
print("=" * 60)

# Test full day layer forward pass including activation
torch.manual_seed(123)
x_test = torch.randn(batch_size, seq_len, neural_dim)
day_idx_test = torch.randint(0, n_days, (batch_size,))

# RNN style
weights_rnn = torch.stack([day_weights_list[i] for i in day_idx_test], dim=0)
biases_rnn = torch.cat([day_biases_list[i] for i in day_idx_test], dim=0).unsqueeze(1)
out_rnn = torch.einsum("btd,bdk->btk", x_test, weights_rnn) + biases_rnn
out_rnn = torch.nn.functional.softsign(out_rnn)

# Conformer style
weights_conf = day_weights_tensor[day_idx_test]
biases_conf = day_biases_tensor[day_idx_test]
out_conf = torch.einsum("btd,bdk->btk", x_test, weights_conf) + biases_conf
out_conf = torch.nn.functional.softsign(out_conf)

full_match = torch.allclose(out_rnn, out_conf)
print(f"[TEST 3] Full forward pass match: {full_match}")

if not full_match:
    print(f"  Max output diff: {(out_rnn - out_conf).abs().max():.2e}")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

all_passed = weights_match and biases_match and grads_match and full_match

if all_passed:
    print("ALL TESTS PASSED")
    print("Day layer implementations are EQUIVALENT")
    print("The indexing change is NOT the cause of poor performance")
else:
    print("SOME TESTS FAILED")
    print("Day layer implementations DIFFER - this could be the bug!")
    print(f"  - Weights match: {weights_match}")
    print(f"  - Biases match: {biases_match}")
    print(f"  - Gradients match: {grads_match}")
    print(f"  - Full forward match: {full_match}")
