#!/usr/bin/env python3
"""convert_weights.py — Convert HuggingFace Qwen2.5-3B safetensors to Annie binary format.

Usage:
    python3 convert_weights.py --model-dir /path/to/Qwen2.5-3B-Instruct --output qwen3b_weights.bin

The HF model directory should contain:
    model-00001-of-00002.safetensors
    model-00002-of-00002.safetensors
    config.json

Output binary layout:
    [AnnieWeightHdr]
    [embed_tokens: vocab_size * dim * fp32]
    Per layer 0..n_layers-1:
        [rms_att: dim * fp32]
        [Wq: dim * dim * fp32]
        [bq: dim * fp32]              (if attention_bias)
        [Wk: kv_dim * dim * fp32]
        [bk: kv_dim * fp32]           (if attention_bias)
        [Wv: kv_dim * dim * fp32]
        [bv: kv_dim * fp32]           (if attention_bias)
        [Wo: dim * dim * fp32]
        [bo: dim * fp32]              (if attention_bias)
        [rms_ffn: dim * fp32]
        [W1: hidden_dim * dim * fp32]
        [W2: dim * hidden_dim * fp32]
        [W3: hidden_dim * dim * fp32]
    [rms_final: dim * fp32]
"""
import argparse
import json
import struct
import numpy as np
from pathlib import Path

def load_safetensors(model_dir):
    """Load all safetensors files into a dict of numpy arrays."""
    try:
        from safetensors import safe_open
    except ImportError:
        print("ERROR: pip install safetensors")
        raise
    try:
        import torch
        USE_TORCH = True
        print("  Using PyTorch backend for bf16 support")
    except ImportError:
        USE_TORCH = False

    weights = {}
    for sf_path in sorted(Path(model_dir).glob("model-*.safetensors")):
        print(f"  Loading {sf_path.name}...")
        if USE_TORCH:
            with safe_open(str(sf_path), framework="pt") as f:
                for key in f.keys():
                    tensor = f.get_tensor(key)
                    weights[key] = tensor.float().numpy()
        else:
            # Manual bf16→fp32: read raw bytes, interpret as uint16, shift to fp32
            from safetensors import safe_open as _so
            import struct as _st
            with open(str(sf_path), 'rb') as raw:
                header_len = _st.unpack('<Q', raw.read(8))[0]
                header = json.loads(raw.read(header_len))
                data_start = 8 + header_len
                for key, meta in header.items():
                    if key == '__metadata__':
                        continue
                    dtype = meta['dtype']
                    shape = meta['data_offsets']
                    begin, end = meta['data_offsets']
                    raw.seek(data_start + begin)
                    raw_bytes = raw.read(end - begin)
                    if dtype == 'BF16':
                        u16 = np.frombuffer(raw_bytes, dtype=np.uint16)
                        u32 = u16.astype(np.uint32) << 16
                        tensor = u32.view(np.float32).reshape(meta['shape'])
                    elif dtype == 'F16':
                        tensor = np.frombuffer(raw_bytes, dtype=np.float16).reshape(meta['shape']).astype(np.float32)
                    elif dtype == 'F32':
                        tensor = np.frombuffer(raw_bytes, dtype=np.float32).reshape(meta['shape'])
                    else:
                        raise ValueError(f"Unsupported dtype {dtype} for {key}")
                    weights[key] = tensor.astype(np.float32)
    return weights


def load_config(model_dir):
    """Load model config from config.json."""
    with open(Path(model_dir) / "config.json") as f:
        cfg = json.load(f)
    return {
        'dim': cfg['hidden_size'],
        'hidden_dim': cfg['intermediate_size'],
        'n_layers': cfg['num_hidden_layers'],
        'n_heads': cfg['num_attention_heads'],
        'n_kv_heads': cfg.get('num_key_value_heads', cfg['num_attention_heads']),
        'head_dim': cfg['hidden_size'] // cfg['num_attention_heads'],
        'vocab_size': cfg['vocab_size'],
        'max_seq_len': 256,  # Training sequence length
        'rope_theta': cfg.get('rope_theta', 1000000.0),
        'rms_norm_eps': cfg.get('rms_norm_eps', 1e-6),
        'attention_bias': cfg.get('attention_bias', True),
        'tie_embeddings': cfg.get('tie_word_embeddings', True),
    }


def write_annie_header(f, cfg):
    """Write AnnieWeightHdr matching C struct layout on ARM64.

    C layout (config.h):
        AnnieConfig (52 bytes with alignment padding):
            int dim,hidden_dim,n_layers       (12)
            int n_heads,n_kv_heads,head_dim   (12)
            int vocab_size,max_seq_len        (8)
            float rope_theta,rms_norm_eps     (8)
            float lora_alpha                  (4)
            int lora_rank                     (4)
            bool tie_embeddings,qkv_bias      (2)
            <2 bytes padding to 4-byte align> (2)

        AnnieWeightHdr (76 bytes):
            int magic,version                 (8)
            AnnieConfig config                (52)
            int pad[4]                        (16)
    """
    MAGIC = 0x414E4E49  # "ANNI"
    VERSION = 1
    # Pack AnnieConfig: 8 ints + 3 floats + 1 int + 2 bools + 2 pad bytes = 52 bytes
    config_bytes = struct.pack('<iiiiiiiifffi??xx',
        cfg['dim'], cfg['hidden_dim'], cfg['n_layers'],
        cfg['n_heads'], cfg['n_kv_heads'], cfg['head_dim'],
        cfg['vocab_size'], cfg['max_seq_len'],
        cfg['rope_theta'], cfg['rms_norm_eps'],
        8.0,   # lora_alpha
        8,     # lora_rank
        cfg['tie_embeddings'], cfg['attention_bias'])
    assert len(config_bytes) == 52, f"AnnieConfig size mismatch: {len(config_bytes)} != 52"

    # AnnieWeightHdr: magic(4) + version(4) + config(52) + pad[4](16) = 76 bytes
    f.write(struct.pack('<ii', MAGIC, VERSION))
    f.write(config_bytes)
    f.write(b'\x00' * 16)  # pad[4]


def main():
    parser = argparse.ArgumentParser(description='Convert Qwen2.5-3B to Annie format')
    parser.add_argument('--model-dir', required=True, help='HuggingFace model directory')
    parser.add_argument('--output', default='qwen3b_weights.bin', help='Output binary file')
    parser.add_argument('--verify', action='store_true', help='Verify by reloading and checking norms')
    args = parser.parse_args()

    print("Loading config...")
    cfg = load_config(args.model_dir)
    print(f"  dim={cfg['dim']} hidden={cfg['hidden_dim']} layers={cfg['n_layers']} "
          f"heads={cfg['n_heads']}/{cfg['n_kv_heads']} vocab={cfg['vocab_size']}")

    kv_dim = cfg['n_kv_heads'] * cfg['head_dim']
    print(f"  kv_dim={kv_dim} head_dim={cfg['head_dim']} rope_theta={cfg['rope_theta']}")
    print(f"  attention_bias={cfg['attention_bias']} tie_embeddings={cfg['tie_embeddings']}")

    print("\nLoading safetensors...")
    weights = load_safetensors(args.model_dir)
    print(f"  Loaded {len(weights)} tensors")

    # Verify shapes
    embed = weights['model.embed_tokens.weight']
    print(f"  embed_tokens: {embed.shape} (expect [{cfg['vocab_size']}, {cfg['dim']}])")
    assert embed.shape == (cfg['vocab_size'], cfg['dim'])

    print(f"\nWriting to {args.output}...")
    with open(args.output, 'wb') as f:
        write_annie_header(f, cfg)

        # Embedding table: [vocab_size, dim] row-major fp32
        embed.tofile(f)
        print(f"  embed_tokens: {embed.shape} ({embed.nbytes / 1e6:.1f} MB)")

        # Per-layer weights
        total_bytes = embed.nbytes
        for L in range(cfg['n_layers']):
            prefix = f'model.layers.{L}'

            # RMSNorm attention
            rms_att = weights[f'{prefix}.input_layernorm.weight']
            rms_att.tofile(f)

            # Q projection: [dim, dim]
            wq = weights[f'{prefix}.self_attn.q_proj.weight']
            assert wq.shape == (cfg['dim'], cfg['dim']), f"Wq shape {wq.shape}"
            wq.tofile(f)
            if cfg['attention_bias']:
                bq = weights[f'{prefix}.self_attn.q_proj.bias']
                bq.tofile(f)

            # K projection: [kv_dim, dim]
            wk = weights[f'{prefix}.self_attn.k_proj.weight']
            assert wk.shape == (kv_dim, cfg['dim']), f"Wk shape {wk.shape}"
            wk.tofile(f)
            if cfg['attention_bias']:
                bk = weights[f'{prefix}.self_attn.k_proj.bias']
                bk.tofile(f)

            # V projection: [kv_dim, dim]
            wv = weights[f'{prefix}.self_attn.v_proj.weight']
            assert wv.shape == (kv_dim, cfg['dim']), f"Wv shape {wv.shape}"
            wv.tofile(f)
            if cfg['attention_bias']:
                bv = weights[f'{prefix}.self_attn.v_proj.bias']
                bv.tofile(f)

            # O projection: [dim, dim]
            wo = weights[f'{prefix}.self_attn.o_proj.weight']
            assert wo.shape == (cfg['dim'], cfg['dim']), f"Wo shape {wo.shape}"
            wo.tofile(f)
            if cfg['attention_bias']:
                bo = weights.get(f'{prefix}.self_attn.o_proj.bias')
                if bo is not None:
                    bo.tofile(f)
                else:
                    # Some Qwen2.5 variants don't have o_proj bias
                    np.zeros(cfg['dim'], dtype=np.float32).tofile(f)

            # RMSNorm FFN
            rms_ffn = weights[f'{prefix}.post_attention_layernorm.weight']
            rms_ffn.tofile(f)

            # FFN weights
            # gate_proj (W1): [hidden_dim, dim]
            w1 = weights[f'{prefix}.mlp.gate_proj.weight']
            assert w1.shape == (cfg['hidden_dim'], cfg['dim']), f"W1 shape {w1.shape}"
            w1.tofile(f)

            # down_proj (W2): [dim, hidden_dim]
            w2 = weights[f'{prefix}.mlp.down_proj.weight']
            assert w2.shape == (cfg['dim'], cfg['hidden_dim']), f"W2 shape {w2.shape}"
            w2.tofile(f)

            # up_proj (W3): [hidden_dim, dim]
            w3 = weights[f'{prefix}.mlp.up_proj.weight']
            assert w3.shape == (cfg['hidden_dim'], cfg['dim']), f"W3 shape {w3.shape}"
            w3.tofile(f)

            layer_bytes = (rms_att.nbytes + wq.nbytes + wk.nbytes + wv.nbytes + wo.nbytes
                          + rms_ffn.nbytes + w1.nbytes + w2.nbytes + w3.nbytes)
            if cfg['attention_bias']:
                layer_bytes += cfg['dim'] * 4 * 3 + kv_dim * 4 * 2  # bq,bo,bk,bv + maybe bo
            total_bytes += layer_bytes
            if L % 6 == 0:
                print(f"  Layer {L}: {layer_bytes / 1e6:.1f} MB")

        # Final RMSNorm
        rms_final = weights['model.norm.weight']
        rms_final.tofile(f)
        total_bytes += rms_final.nbytes

    print(f"\nDone! {args.output}: {total_bytes / 1e9:.2f} GB")

    if args.verify:
        print("\nVerification: reloading and checking tensor norms...")
        data = np.fromfile(args.output, dtype=np.float32)
        # Skip header (exact: 76 bytes = 19 floats)
        header_size = 76
        offset = header_size // 4
        # Check embed norm
        embed_loaded = data[offset:offset + cfg['vocab_size'] * cfg['dim']].reshape(embed.shape)
        norm_diff = abs(np.linalg.norm(embed_loaded) - np.linalg.norm(embed.flatten()))
        print(f"  embed norm diff: {norm_diff:.6f} (should be ~0)")
        if norm_diff > 0.01:
            print("  WARNING: norm mismatch!")
        else:
            print("  Verification passed!")


if __name__ == '__main__':
    main()
