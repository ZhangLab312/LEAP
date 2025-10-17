DNA_LM from Performer

DNA_LM{
    num_tokens=CLASS,
    dim=200,
    depth=8,
    max_seq_len=SEQ_LEN,
    heads=10,
    local_attn_heads=0
}
GPU:4090
learning rate within the range of {1e-4, 5e-5, 1e-5} and the weight decay within the range of {0.01, 0.001}.