from EEG import DDIMScheduler
noise_scheduler=dict(
        type=DDIMScheduler,
        beta_end=0.02,
        beta_schedule="linear",
        beta_start=0.0001,
        clip_sample=True,
        num_train_timesteps=200,
        prediction_type="epsilon",
        set_alpha_to_one=True,
        steps_offset=0,
        trained_betas=None
    )