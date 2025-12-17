feature_coords = {
    "atac": {
        "chrom_ids": chrom_ids_long,   # (F,)
        "start": start_bp,             # (F,)
        "end": end_bp,                 # (F,)
    }
}

attn_bias_cfg = {
    "atac": {
        "type": "distance",
        "lengthscale_bp": 50_000.0,
        "same_chrom_only": True,
    }
}

trainer = UniVITrainer(
    model,
    train_loader,
    val_loader=val_loader,
    train_cfg=TrainingConfig(...),
    device="cuda",
    feature_coords=feature_coords,
    attn_bias_cfg=attn_bias_cfg,
)
trainer.fit()

