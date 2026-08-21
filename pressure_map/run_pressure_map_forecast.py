#!/usr/bin/env python3
"""Run the published causal pressure-map forecasting stage.

This command runs WeatherNext Cyclones Mini to generate a global weather
rollout and writes a compact Pacific pressure-map packet. It does not download
or consume JMA, JTWC, ECMWF, GFS, or GEFS forecast products.

The initial state must contain exactly two analysis snapshots, t-6 h and t0.
The template is a WeatherNext-compatible shape/configuration sample. Its
time-varying values are replaced from the initial state before inference; it
is never used as a source of future weather inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from datetime import timedelta
from pathlib import Path


MODEL_NAME = "WeatherNextCyclones_Mini"
MODEL_SPLIT = "2024"
CONFIG_NAME = f"weathernext2/configs/{MODEL_NAME}"
ALLOWED_FORCINGS = {
    "year_progress_sin",
    "year_progress_cos",
    "day_progress_sin",
    "day_progress_cos",
}
SELECTED_OUTPUTS = (
    "mean_sea_level_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "sea_surface_temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "geopotential",
    "cyclone_exists_gaussian_unit_mode",
    "cyclone_all_wind_disc",
    "cyclone_usa_wind_disc",
    "cyclone_usa_pres_disc",
    "cyclone_usa_rmw_disc",
    "cyclone_usa_r34_ne_radius_disc",
    "cyclone_usa_r34_se_radius_disc",
    "cyclone_usa_r34_sw_radius_disc",
    "cyclone_usa_r34_nw_radius_disc",
    "cyclone_usa_r50_ne_radius_disc",
    "cyclone_usa_r50_se_radius_disc",
    "cyclone_usa_r50_sw_radius_disc",
    "cyclone_usa_r50_nw_radius_disc",
    "cyclone_usa_r64_ne_radius_disc",
    "cyclone_usa_r64_se_radius_disc",
    "cyclone_usa_r64_sw_radius_disc",
    "cyclone_usa_r64_nw_radius_disc",
)
PRESSURE_MAP_REGION = {"lat": (0.0, 45.0), "lon": (100.0, 220.0)}
PRESSURE_MAP_STRIDE = 2
ADAPTER_PRESSURE_LEVELS_HPA = (200, 500, 700, 850)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        required=True,
        type=Path,
        help="WeatherNext Cyclones Mini checkpoint (.npz).",
    )
    parser.add_argument(
        "--template",
        required=True,
        type=Path,
        help="WeatherNext-compatible sample/template NetCDF.",
    )
    parser.add_argument(
        "--initial-state",
        required=True,
        type=Path,
        help="Two-time causal analysis NetCDF containing t-6 h and t0.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=20, help="6-hour steps, 1..20.")
    parser.add_argument("--members", type=int, default=1, help="Same-model members.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--jax-cache",
        type=Path,
        help="Optional persistent JAX compilation cache directory.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _analysis_times(state, pd):
    """Read the two analysis times from common WeatherNext state layouts."""
    coordinate = state.coords.get("datetime", state.coords.get("time"))
    if coordinate is None:
        raise ValueError("Initial state needs a datetime or time coordinate")
    values = coordinate.values
    if getattr(coordinate, "dims", ()) and "batch" in coordinate.dims:
        coordinate = coordinate.isel(batch=0)
        values = coordinate.values
    values = values.reshape(-1)
    if len(values) != 2:
        raise ValueError(f"Initial state must expose exactly two times, got {values}")
    return [pd.Timestamp(value) for value in values]


def inject_initial_state(template, initial_state_path: Path, task, xr, np, pd):
    """Replace all time-varying template inputs with the causal two-state file."""
    if not initial_state_path.is_file():
        raise FileNotFoundError(initial_state_path)
    state = xr.load_dataset(initial_state_path, engine="h5netcdf").compute()
    if state.sizes.get("time") != 2:
        raise ValueError("Initial state must contain exactly two weather times")
    state_times = _analysis_times(state, pd)
    if state_times[1] - state_times[0] != timedelta(hours=6):
        raise ValueError(f"Initial-state spacing is not six hours: {state_times}")
    init_time = pd.Timestamp(state_times[-1])

    if template.sizes.get("batch") != 1:
        raise ValueError("The public runner currently requires a one-batch template")
    for coord in ("lat", "lon", "level"):
        if coord in state.coords and coord in template.coords:
            if not np.array_equal(state[coord].values, template[coord].values):
                raise ValueError(f"Initial-state {coord} coordinate mismatches template")

    output = template.copy(deep=False)
    timeline = np.asarray(
        [
            np.datetime64(
                (init_time - timedelta(hours=6) + timedelta(hours=6 * i)).to_datetime64(),
                "ns",
            )
            for i in range(output.sizes["time"])
        ],
        dtype="datetime64[ns]",
    )
    output = output.assign_coords(datetime=(("batch", "time"), timeline[None]))
    for name in ALLOWED_FORCINGS | {"seconds_since_epoch"}:
        if name in output:
            output = output.drop_vars(name)

    input_variables = set(task.input_variables) - ALLOWED_FORCINGS
    missing = input_variables - set(state.data_vars)
    if missing:
        raise RuntimeError(f"Initial state is missing model inputs: {sorted(missing)}")

    for name in input_variables:
        source = state[name]
        target = output[name]
        if "batch" in source.dims:
            source = source.isel(batch=0, drop=True)
        if "time" in target.dims:
            source = source.transpose(*[dim for dim in target.dims if dim != "batch"])
            values = np.full(target.shape, np.nan, dtype="float32")
            values[:, :2] = source.values[None]
            output[name] = xr.DataArray(
                values,
                dims=target.dims,
                coords={dim: target.coords[dim] for dim in target.dims if dim in target.coords},
            )
        else:
            source = source.transpose(*target.dims)
            output[name] = xr.DataArray(
                source.values.astype("float32"),
                dims=target.dims,
                coords={dim: target.coords[dim] for dim in target.dims if dim in target.coords},
            )

    # Target-only template values are never inputs. NaN makes that boundary
    # visible in the serialized input/output provenance.
    for name in set(task.target_variables) - input_variables:
        if name in output and "time" in output[name].dims:
            output[name] = xr.full_like(output[name], np.nan)
    output.attrs["weather_initialization"] = str(initial_state_path.resolve())
    output.attrs["latest_weather_input_utc"] = init_time.isoformat()
    output.attrs["future_weather_inputs_used"] = False
    return output, init_time


def configure_attention(config: object, backend: str) -> str:
    transformer_kwargs = config.predictor_kwargs["noisy_function_kwargs"][
        "mesh_model_ctor"
    ].keywords["transformer_kwargs"]
    original = transformer_kwargs["attention_type"]
    if backend != "tpu":
        transformer_kwargs["attention_type"] = "triblockdiag_mha"
    else:
        transformer_kwargs.update(
            {
                "block_q": 128,
                "block_kv": 128,
                "block_kv_compute": 128,
                "block_q_dkv": 128,
                "block_kv_dkv": 128,
                "block_kv_dkv_compute": 128,
            }
        )
    return f"{original}->{transformer_kwargs['attention_type']}"


def save_selected_fields(predictions, path: Path, np):
    selected = predictions[[name for name in SELECTED_OUTPUTS if name in predictions]]
    if "level" in selected.coords:
        available = set(int(value) for value in selected.level.values)
        levels = [level for level in ADAPTER_PRESSURE_LEVELS_HPA if level in available]
        if levels:
            selected = selected.sel(level=levels)
    selected = selected.astype(np.float32)
    encoding = {
        name: {"compression": "gzip", "compression_opts": 2, "shuffle": True}
        for name in selected.data_vars
    }
    selected.to_netcdf(path, engine="h5netcdf", encoding=encoding)


def _regional_values(data, lat_index, lon_index, np, level=None):
    if level is not None:
        matches = np.flatnonzero(np.asarray(data.level.values) == level)
        if len(matches) != 1:
            raise ValueError(f"Level {level} is missing from {data.name}")
        data = data.isel(level=int(matches[0]))
    if "batch" in data.dims:
        data = data.isel(batch=0, drop=True)
    if "sample" not in data.dims:
        data = data.expand_dims(sample=[0])
    data = data.transpose("sample", "time", "lat", "lon")
    return np.asarray(data.values, dtype="float32")[:, :, lat_index][:, :, :, lon_index]


def write_pressure_map_packet(predictions, output_dir: Path, issue_time, np, json, xr):
    """Write portable regional maps plus a manifest for downstream renderers."""
    lat = np.asarray(predictions.lat.values, dtype="float32")
    lon = np.asarray(predictions.lon.values, dtype="float32")
    lat_index = np.flatnonzero(
        (lat >= PRESSURE_MAP_REGION["lat"][0])
        & (lat <= PRESSURE_MAP_REGION["lat"][1])
    )[::PRESSURE_MAP_STRIDE]
    lon_index = np.flatnonzero(
        (lon >= PRESSURE_MAP_REGION["lon"][0])
        & (lon <= PRESSURE_MAP_REGION["lon"][1])
    )[::PRESSURE_MAP_STRIDE]
    if not len(lat_index) or not len(lon_index):
        raise ValueError("WeatherNext output does not cover the Pacific map region")

    fields = []

    def add(name, source_name, level=None, scale=1.0, offset=0.0):
        if source_name not in predictions:
            fields.append((name, np.full((predictions.sizes["sample"], predictions.sizes["time"], len(lat_index), len(lon_index)), np.nan, dtype="float32")))
            return
        values = _regional_values(predictions[source_name], lat_index, lon_index, np, level)
        values = values * scale + offset
        fields.append((name, values.astype("float32")))

    add("mslp_hpa", "mean_sea_level_pressure", scale=0.01)
    add("sst_k", "sea_surface_temperature")
    add("u10_ms", "10m_u_component_of_wind")
    add("v10_ms", "10m_v_component_of_wind")
    for level in (850, 500, 200):
        add(f"u{level}_ms", "u_component_of_wind", level=level)
        add(f"v{level}_ms", "v_component_of_wind", level=level)
    add("h500_m", "geopotential", level=500, scale=1.0 / 9.80665)
    add("cyclone_risk", "cyclone_exists_gaussian_unit_mode")
    add("cyclone_all_wind_disc", "cyclone_all_wind_disc")
    add("cyclone_usa_wind_disc", "cyclone_usa_wind_disc")
    add("cyclone_usa_pres_disc", "cyclone_usa_pres_disc")
    add("cyclone_usa_rmw_disc", "cyclone_usa_rmw_disc")
    for radius in ("r34", "r50", "r64"):
        for quadrant in ("ne", "se", "sw", "nw"):
            add(
                f"cyclone_usa_{radius}_{quadrant}_radius_disc",
                f"cyclone_usa_{radius}_{quadrant}_radius_disc",
            )

    names = [name for name, _ in fields]
    maps = np.stack([values for _, values in fields], axis=2)
    np.savez_compressed(
        output_dir / "pressure_maps.npz",
        maps=maps.astype("float32"),
        field_names=np.asarray(names),
        lat=lat[lat_index],
        lon=lon[lon_index],
        lead_hours=np.arange(6, 6 * predictions.sizes["time"] + 1, 6, dtype="int16"),
    )
    manifest = {
        "schema_version": 1,
        "artifact": "Trackformer 1.2 published pressure-map forecasting stage",
        "backend": MODEL_NAME,
        "checkpoint_split": MODEL_SPLIT,
        "issue_time_utc": issue_time.isoformat(),
        "lead_hours": list(range(6, 6 * predictions.sizes["time"] + 1, 6)),
        "members": int(predictions.sizes["sample"]),
        "map_region": PRESSURE_MAP_REGION,
        "grid_stride_degrees": PRESSURE_MAP_STRIDE,
        "maps_shape": list(maps.shape),
        "fields": names,
        "causal_policy": {
            "initial_weather_times": ["t-6 h", "t0"],
            "future_weather_inputs_used": False,
            "official_forecast_inputs_used": False,
            "official_forecasts_are_not_model_inputs": True,
            "generated_fields_are_model_outputs": True,
        },
        "outputs": {
            "predictions_selected": "predictions_selected.nc",
            "compact_maps": "pressure_maps.npz",
        },
    }
    (output_dir / "pressure_map_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return manifest


def main() -> None:
    args = parse_args()
    if not 1 <= args.steps <= 20:
        raise ValueError("--steps must be in [1, 20]")
    if args.members < 1:
        raise ValueError("--members must be positive")
    for path in (args.weights, args.template, args.initial_state):
        if not path.is_file():
            raise FileNotFoundError(path)

    # Keep heavyweight imports inside main so --help and static validation work
    # without installing the accelerator stack.
    import dataclasses
    import os

    import haiku as hk
    import jax
    import numpy as np
    import pandas as pd
    import xarray as xr
    import xarray_jax
    from weathernext.utils import checkpoint, data_utils, fiddle_config_io, rollout
    from weathernext.weathernext2 import fgn

    if args.jax_cache is not None:
        args.jax_cache.mkdir(parents=True, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", str(args.jax_cache))
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    backend = jax.default_backend()
    devices = jax.local_devices()
    if args.members % len(devices):
        raise ValueError(
            f"--members ({args.members}) must be a multiple of local devices ({len(devices)})"
        )

    config = fiddle_config_io.get_fiddle_config_by_name(CONFIG_NAME)
    attention = configure_attention(config, backend)
    with args.weights.open("rb") as handle:
        ckpt = checkpoint.load(handle, fgn.CheckPoint)
    template = xr.load_dataset(args.template, engine="h5netcdf").compute()
    if "batch" not in template.dims:
        template = template.expand_dims(batch=[0])
    example_batch, init_time = inject_initial_state(
        template, args.initial_state, config.task, xr, np, pd
    )
    forcing_variables = set(config.task.forcing_variables)
    if forcing_variables != ALLOWED_FORCINGS:
        raise RuntimeError(f"Unexpected future forcing variables: {sorted(forcing_variables)}")
    eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
        example_batch,
        target_lead_times=slice("6h", f"{args.steps * 6}h"),
        **dataclasses.asdict(config.task),
    )
    input_hours = eval_inputs.time.values.astype("timedelta64[h]").astype(int)
    target_hours = eval_targets.time.values.astype("timedelta64[h]").astype(int)
    if input_hours.max() > 0 or target_hours.min() <= 0:
        raise RuntimeError(
            f"Causal boundary failed: input lead hours={input_hours.tolist()}, "
            f"target lead hours={target_hours.tolist()}"
        )

    config_inference = fgn.PredictorConfig(
        task=config.task,
        predictor_constructor=config.predictor_constructor,
        predictor_kwargs=config.predictor_kwargs,
        predictor_wrappers=config.predictor_wrappers[:-1],
    )

    @hk.transform
    def run_forward(inputs, targets_template, forcings):
        predictor = fgn.construct_predictor(config_inference)
        return predictor(inputs, targets_template=targets_template, forcings=forcings)

    run_forward_jitted = jax.jit(
        lambda rng, inputs, template, forcings: run_forward.apply(
            ckpt.params, rng, inputs, template, forcings
        )
    )
    run_forward_pmap = xarray_jax.pmap(run_forward_jitted, dim="sample")
    seed = jax.random.PRNGKey(args.seed)
    rngs = np.stack([jax.random.fold_in(seed, i) for i in range(args.members)], axis=0)
    print(
        f"backend={backend} devices={devices} members={args.members} steps={args.steps} "
        f"attention={attention}",
        flush=True,
    )
    chunks = []
    for index, chunk in enumerate(
        rollout.chunked_prediction_generator_multiple_runs(
            predictor_fn=run_forward_pmap,
            rngs=rngs,
            inputs=eval_inputs,
            targets_template=eval_targets * np.nan,
            forcings=eval_forcings,
            num_steps_per_chunk=1,
            num_samples=args.members,
            pmap_devices=devices,
        ),
        start=1,
    ):
        if index > args.steps:
            break
        chunks.append(jax.device_get(chunk))
        print(f"rollout step={index}/{args.steps} elapsed_s={time.time() - started:.1f}", flush=True)
    if not chunks:
        raise RuntimeError("WeatherNext returned no forecast chunks")
    predictions = xr.combine_by_coords(chunks).isel(time=slice(0, args.steps))
    if predictions.sizes.get("time") != args.steps:
        raise RuntimeError("Rollout did not produce the requested horizon")
    if predictions.sizes.get("sample") != args.members:
        raise RuntimeError("Rollout did not preserve the requested member count")

    fields_path = args.output / "predictions_selected.nc"
    save_selected_fields(predictions, fields_path, np)
    pressure_manifest = write_pressure_map_packet(
        predictions, args.output, init_time, np, json, xr
    )
    manifest = {
        **pressure_manifest,
        "weights": {"path": str(args.weights.resolve()), "sha256": sha256(args.weights)},
        "template": {"path": str(args.template.resolve()), "sha256": sha256(args.template)},
        "initial_state": {"path": str(args.initial_state.resolve()), "sha256": sha256(args.initial_state)},
        "input_policy": {
            "latest_weather_input_lead_hours": int(input_hours.max()),
            "weather_input_lead_hours": input_hours.tolist(),
            "future_forcings": sorted(forcing_variables),
            "future_weather_targets_used_as_inputs": False,
            "official_forecast_inputs_used": False,
        },
        "runtime": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "backend": backend,
            "devices": [str(device) for device in devices],
            "seconds": time.time() - started,
        },
    }
    (args.output / "pressure_map_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    import os

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    main()
