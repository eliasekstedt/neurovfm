def gradient_smoke_test_etc(batch):
    # --------------------------------------------------
    # Gradient smoke test
    # --------------------------------------------------

    batch["img"] = batch["img"].clone().requires_grad_(True)

    logit = model(batch)

    print("\n==== OUTPUT ====")
    print("logit:", logit)
    print("shape:", logit.shape)
    print("requires_grad:", logit.requires_grad)

    model.zero_grad()

    logit.backward()

    print("\n==== INPUT GRAD ====")
    print("grad is None:", batch["img"].grad is None)

    if batch["img"].grad is not None:
        print("grad shape:", batch["img"].grad.shape)
        print("grad dtype:", batch["img"].grad.dtype)
        print("grad abs sum:", batch["img"].grad.abs().sum())
        print("grad abs max:", batch["img"].grad.abs().max())
        print("grad abs mean:", batch["img"].grad.abs().mean())

    # --------------------------------------------------
    # Device / dtype sanity
    # --------------------------------------------------

    print("\n==== MODEL ====")
    print("ann dtype:", next(model.classifier.parameters()).dtype)

    # --------------------------------------------------
    # NeuroVFM output
    # --------------------------------------------------

    with torch.enable_grad():
        tokens = model.encoder.embed(batch)

    print("\n==== TOKENS ====")
    print("shape:", tokens.shape)
    print("dtype:", tokens.dtype)
    print("requires_grad:", tokens.requires_grad)

    # --------------------------------------------------
    # Input stats
    # --------------------------------------------------

    print("\n==== INPUT ====")
    print("img shape:", batch["img"].shape)
    print("img dtype:", batch["img"].dtype)
    print("img min:", batch["img"].min())
    print("img max:", batch["img"].max())

    # --------------------------------------------------
    # Coordinates
    # --------------------------------------------------

    print("\n==== COORDS ====")
    print("coords shape:", batch["coords"].shape)
    print("coords min:", batch["coords"].min(dim=0).values)
    print("coords max:", batch["coords"].max(dim=0).values)

def broken_grad_test(batch, model):
    batch["img"] = batch["img"].clone().requires_grad_(True)

    tokens = model.encoder.embed(batch)

    print("tokens requires_grad:", tokens.requires_grad)
    print("tokens grad_fn:", tokens.grad_fn)

    vec = tokens.mean(dim=0)

    print("vec requires_grad:", vec.requires_grad)
    print("vec grad_fn:", vec.grad_fn)

    vec = vec.float().unsqueeze(0)

    logits = model.classifier(vec)

    print("logits requires_grad:", logits.requires_grad)
    print("logits grad_fn:", logits.grad_fn)

def final_diagnostic_lol(batch, encoder):
    tokens = batch["img"].clone().requires_grad_(True)

    print("input requires_grad:", tokens.requires_grad)

    tokens2 = encoder.norm_module.normalize(
        tokens,
        batch["mode"],
        batch["path"],
        cu_seqlens=batch["series_cu_seqlens"].to(encoder.device),
        sizes=batch["size"]
    )

    print("normalized requires_grad:", tokens2.requires_grad)
    print("normalized grad_fn:", tokens2.grad_fn)

def f0(batch, encoder, model):
    import torch
    tokens = batch["img"].to(encoder.device).clone().requires_grad_(True)
    coords = batch["coords"].to(encoder.device)
    cu_seqlens = batch["series_cu_seqlens"].to(encoder.device)

    tokens_norm = encoder.norm_module.normalize(
        tokens,
        batch["mode"],
        batch["path"],
        cu_seqlens=cu_seqlens,
        sizes=batch["size"],
    )

    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        embs = encoder.model(
            tokens_norm,
            coords,
            masks=None,
            cu_seqlens=cu_seqlens,
            max_seqlen=batch["series_max_len"],
            use_flash_attn=False,
        )

    vec = embs.mean(dim=0).float().unsqueeze(0)
    logit = model.classifier(vec)

    logit.backward()

    print("embs requires_grad:", embs.requires_grad)
    print("logit requires_grad:", logit.requires_grad)
    print("input grad exists:", tokens.grad is not None)
    print("input grad abs sum:", tokens.grad.abs().sum())


def f1(batch, encoder, model):
    import torch
    print("before:")
    print("grad enabled:", torch.is_grad_enabled())
    print("inference mode:", torch.is_inference_mode_enabled())

    with torch.inference_mode(False), torch.enable_grad():
        x = batch["img"].detach().clone().to(encoder.device).requires_grad_(True)
        coords = batch["coords"].to(encoder.device)
        cu_seqlens = batch["series_cu_seqlens"].to(encoder.device)

        x_norm = encoder.norm_module.normalize(
            x,
            batch["mode"],
            batch["path"],
            cu_seqlens=cu_seqlens,
            sizes=batch["size"],
        )

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            embs = encoder.model(
                x_norm,
                coords,
                masks=None,
                cu_seqlens=cu_seqlens,
                max_seqlen=batch["series_max_len"],
                use_flash_attn=False,
            )

        vec = embs.mean(dim=0).float().unsqueeze(0)
        logit = model.classifier(vec)

        print("\ninside:")
        print("grad enabled:", torch.is_grad_enabled())
        print("inference mode:", torch.is_inference_mode_enabled())
        print("x:", x.requires_grad, x.is_leaf)
        print("x_norm:", x_norm.requires_grad, x_norm.grad_fn)
        print("embs:", embs.requires_grad, embs.grad_fn)
        print("logit:", logit.requires_grad, logit.grad_fn)

        logit.sum().backward()

        print("\ngradient:")
        print("exists:", x.grad is not None)
        if x.grad is not None:
            print("shape:", x.grad.shape)
            print("absolute sum:", x.grad.abs().sum().item())
            print("absolute max:", x.grad.abs().max().item())