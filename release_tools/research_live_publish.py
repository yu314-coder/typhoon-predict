"""Stage and publish completed TrackFormer research issues to a data-only branch."""
from argparse import ArgumentParser
from array import array
from datetime import datetime, timedelta, timezone
import gzip
from hashlib import sha256
from io import BytesIO
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


ROOT = "/api/research-live"
BRANCH = "trackformer-research-live-data"
REMOTE = "https://github.com/yu314-coder/typhoon-predict.git"
PRODUCTS = {"pressure": "1375", "temperature": "1375", "rainfall": "v59"}
LEADS = list(range(6, 121, 6))
HASH = re.compile(r"[0-9a-f]{64}")
ISO = re.compile(r"\d{4}-\d{2}-\d{2}T(?:00|06|12|18):00:00Z")
SLOT = re.compile(r"\d{8}T(?:00|06|12|18)Z")


def read(path):
    return json.loads(path.read_bytes())


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode()


def hexdigest(data):
    return sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def issue_time(value):
    require(isinstance(value, str) and ISO.fullmatch(value), "Expected a six-hour UTC cycle.")
    when = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(when <= datetime.now(timezone.utc), "Issue time is in the future.")
    return when


def formatted(when):
    return when.strftime("%Y-%m-%dT%H:%M:%SZ")


def check_inputs(receipt, product, when):
    require(receipt.get("schema") == 1 and receipt.get("product") == product
            and receipt.get("issue_utc") == formatted(when)
            and receipt.get("inference_only") is True
            and receipt.get("external_forecast_inputs") is False
            and receipt.get("supplied_storm_position_or_motion") is False, "Missing causal inference receipt.")
    records = receipt.get("hour_zero_inputs", [])
    require(isinstance(records, list) and 9 <= len(records) <= 64, "Expected the issue plus eight past analyses.")
    allowed = {formatted(when - timedelta(hours=6 * i)) for i in range(9)}
    require(all(row["issue_utc"] in allowed and row["source"] == "NOAA GFS"
                and row["forecast_hour"] == 0 and HASH.fullmatch(row["sha256"]) for row in records)
            and {row["issue_utc"] for row in records} == allowed, "Invalid hour-zero source, timestamp or hash.")
    if product == "temperature":
        require(receipt.get("fine_step") == "1.3.79" and receipt.get("parent_issue_utc") == formatted(when),
                "Temperature requires the coeval 1.3.75 parent and 1.3.79 fine step.")
    if product == "rainfall":
        require(receipt.get("recent_satellite_rain") is False and receipt.get("rain_draws") == 8,
                "Expected V59 placement of eight members without live satellite rain.")


def check_packet(frame, grid, payload):
    require(0 < len(payload) < 2 * 1024 * 1024 and hexdigest(payload) == frame.get("gzip_sha256"),
            "Invalid compressed frame.")
    rows, cols = grid["rows"], grid["columns"]
    n = rows * cols
    masked = frame.get("codec") == "f32-mask-xor-shuffle"
    require(frame.get("codec") in ("f32-xor-shuffle", "f32-mask-xor-shuffle"), "Unexpected frame codec.")
    size = n * (5 if masked else 4)
    require(frame.get("bytes") == size and (frame.get("layout") == "f32-mask") == masked,
            "Invalid frame layout or size.")
    with gzip.GzipFile(fileobj=BytesIO(payload)) as stream:
        encoded = stream.read(size + 1)
    require(len(encoded) == size and hexdigest(encoded) == frame.get("encoded_sha256"),
            "Invalid encoded frame.")
    body = bytearray(n * 4)
    for plane in range(4):
        body[plane::4] = encoded[plane * n:(plane + 1) * n]
    words = array("I")
    words.frombytes(body)
    require(words.itemsize == 4, "Expected IEEE 32-bit float frames.")
    for y in range(0, n, cols):
        for x in range(y + 1, y + cols):
            words[x] ^= words[x - 1]
    raw = words.tobytes()
    mask = encoded[n * 4:]
    require(hexdigest(raw + mask) == frame.get("sha256"), "Decoded frame hash mismatch.")
    fields = memoryview(raw).cast("f")
    if masked:
        require(mask.count(1) == frame.get("valid_count") and 0 < mask.count(1) <= n
                and all(bit in (0, 1) and math.isfinite(v) == bool(bit) for bit, v in zip(mask, fields)),
                "Invalid field support mask.")
    else:
        require(all(math.isfinite(v) for v in fields), "Unmasked nonfinite field.")


def model_specs(product):
    dims = (240, 320) if product == "rainfall" else (241, 321)
    grid = dict(rows=dims[0], columns=dims[1], north=60, south=0, west=100, east=180,
                sample_kind="cells" if product == "rainfall" else "nodes", latitude_order="north-to-south")
    field, units = {"pressure": ("pressure", "hPa"), "temperature": ("t2m", "K"),
                    "rainfall": ("rainfall", "mm / 6 h")}[product]
    return grid, field, units


def stage(product, catalog_path, packet_root, spool, receipt, saved_import):
    source = read(catalog_path)
    when = issue_time(source["issue_utc"])
    cycle = when.strftime("%Y%m%dT%HZ")
    require(source["leads"] == LEADS, "The source must contain all 20 leads.")
    valid = [formatted(when + timedelta(hours=lead)) for lead in LEADS]
    require(source["valid_utc"] == valid, "Valid times do not match this issue.")
    if not saved_import:
        check_inputs(read(receipt), product, when)
    fields, shape, objects = [], model_specs(product), {}
    grid, field, units = shape
    for key in (PRODUCTS[product], "gfs"):
        model = next(row for row in source["models"] if row["key"] == key)
        require(model.get("issue_utc", source["issue_utc"]) == source["issue_utc"], "Mixed forecast cycles.")
        data = model.get("fields", {}).get("t2m", model) if product == "temperature" else model
        require(data["grid"] == grid and data["units"] == units and len(data["frames"]) == 20,
                "Unexpected native field or support.")
        candidate = key != "gfs"
        label = "TrackFormer 1.3.75" if key == "1375" else "Rain V59" if key == "v59" else (
            "NOAA GFS · 0.5 degree" if product == "rainfall" else "NOAA GFS")
        members = 8 if key == "v59" else 1
        require(model["members"] == members, "Unexpected forecast member count.")
        native_grid = {
            "pressure": "2.5 degree model, drawn on a finer grid",
            "temperature": "1.3.79 fine land step",
            "rainfall": "0.25° rainfall cells",
        }[product] if candidate else ("0.5° rainfall cells · no smoothing" if product == "rainfall"
                                     else "Saved GFS field")
        statistic = "NOAA GFS forecast · comparison only" if not candidate else {
            "pressure": "Single forecast · " + native_grid,
            "temperature": "2 m temperature · 1.3.79 fine land step inside 0–60°N, 100–180°E",
            "rainfall": "Placed map from eight drawn rain maps",
        }[product]
        entry = dict(key=key, label=label, units=units, members=members, statistic=statistic,
                     native_grid=native_grid, issue_utc=source["issue_utc"], grid=grid, field=field, frames=[])
        if candidate:
            require(HASH.fullmatch(model.get("checkpoint_sha256", "")), "Missing model checkpoint hash.")
            entry["checkpoint_sha256"] = model["checkpoint_sha256"]
            entry["calibration"] = {
                "pressure": "1.3.71 weather fields with the 1.3.74 central-pressure step where available. "
                            "The official central pressure at issue time can be given; position and motion are inferred.",
                "temperature": "1.3.79 uses the 1.3.75 background and GFS current-weather files (hour zero) "
                               "for temperature over the last 18 hours, soil water and snow.",
                "rainfall": "V58 E2 plus V59 placement. No storm input or recent satellite rain. "
                            "Fine fields come from the GFS current-weather file (hour zero).",
            }[product]
        if product == "pressure" and "track" in model:
            points = model["track"]
            require(len(points) == 20 and all(point is None or (
                len(point) == 3 and all(math.isfinite(x) for x in point)
                and -90 <= point[0] <= 90 and -180 <= point[1] <= 360) for point in points),
                    "Invalid source route.")
            entry["track"] = points
            entry["track_valid"] = [point is not None for point in points]
            if model.get("start") is not None:
                start = model["start"]
                require(len(start) == 2 and all(math.isfinite(x) for x in start), "Invalid own start.")
                entry["start"] = start
        for packet in data["frames"]:
            url = packet["url"]
            require(re.fullmatch(r"/data/forecast-comparison/[\w/.-]+\.bin\.gz", url)
                    and ".." not in Path(url).parts, "Invalid source packet path.")
            path = (packet_root / url[1:]).resolve()
            require(path.is_relative_to(packet_root.resolve()), "Source packet escaped the input directory.")
            payload = path.read_bytes()
            check_packet(packet, grid, payload)
            name = packet["gzip_sha256"] + ".bin.gz"
            spec = {k: packet[k] for k in ("sha256", "gzip_sha256", "encoded_sha256", "bytes",
                                           "codec", "layout", "valid_count") if k in packet}
            entry["frames"].append(dict(url=f"{ROOT}/{product}/{cycle}/{name}", **spec))
            objects[name] = payload
        fields.append(entry)
    note = ("Forecast inputs come from GFS current-weather files (hour zero). "
            "NOAA GFS and agency forecast tracks are comparison only.")
    if saved_import:
        note += " This is the original saved issue."
    catalog = dict(schema=1, issue_utc=source["issue_utc"], valid_utc=valid, leads=LEADS, models=fields,
                   product=product, note=note)
    payload = json_bytes(catalog)
    digest = hexdigest(payload)
    origin = "saved-import" if saved_import else "inference"
    pointer = dict(schema=1, product=product, issue_utc=source["issue_utc"], catalog_sha256=digest, origin=origin)
    dest = spool / product / cycle
    if dest.exists():
        require(read(dest / "metadata.json") == pointer, "An issue with different bytes is already staged.")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".stage-", dir=dest.parent) as tmp:
        ready = Path(tmp) / "complete"
        ready.mkdir()
        for name, body in objects.items():
            (ready / name).write_bytes(body)
        (ready / f"catalog-{digest}.json").write_bytes(payload)
        (ready / "metadata.json").write_bytes(json_bytes(pointer))
        if not saved_import:
            (ready / "input-receipt.sha256").write_text(hexdigest(receipt.read_bytes()) + "\n")
        os.rename(ready, dest)
    print(f"Staged {product} {source['issue_utc']} with {len(objects)} verified packets.")


def git(cwd, *args, check=True):
    config = ("-c", "credential.helper=", "-c", "credential.helper=!gh auth git-credential",
              "-c", "user.name=TrackFormer research forecasts", "-c", "user.email=")
    return subprocess.run(["git", *config, *args], cwd=cwd, text=True, check=check, capture_output=True)


def output_checkout(checkout):
    checkout.mkdir(parents=True, exist_ok=True)
    if not (checkout / ".git").exists():
        require(not any(checkout.iterdir()), "The data checkout must start empty.")
        git(checkout, "init", "-b", BRANCH)
        git(checkout, "remote", "add", "origin", REMOTE)
        git(checkout, "config", "trackformer.researchFeed", "true")
    require(git(checkout, "config", "--get", "trackformer.researchFeed", check=False).stdout.strip() == "true"
            and git(checkout, "remote", "get-url", "origin").stdout.strip() == REMOTE
            and git(checkout, "symbolic-ref", "--short", "HEAD").stdout.strip() == BRANCH,
            "The named checkout is not the separate research data branch.")
    require(not git(checkout, "status", "--porcelain").stdout.strip(), "The data checkout has uncommitted changes.")
    result = git(checkout, "ls-remote", "--exit-code", "origin", f"refs/heads/{BRANCH}", check=False)
    require(result.returncode in (0, 2), "Could not read the research data branch.")
    if result.returncode == 0:
        git(checkout, "fetch", "--quiet", "origin", f"refs/heads/{BRANCH}")
        if git(checkout, "rev-parse", "--verify", "HEAD", check=False).returncode == 0:
            git(checkout, "merge", "--ff-only", "FETCH_HEAD")
        else:
            git(checkout, "switch", "-C", BRANCH, "FETCH_HEAD")
    return result.stdout.split()[0] if result.returncode == 0 else None


def publish(spool, checkout, dry_run, quiet):
    # Validate everything before mutating the data checkout.
    uploads = []
    for product in PRODUCTS:
        candidates = sorted((spool / product).glob("*/metadata.json"), reverse=True)
        if not candidates:
            continue
        metadata_file = candidates[0]
        cycle = metadata_file.parent.name
        pointer = read(metadata_file)
        digest = pointer.get("catalog_sha256", "")
        when = issue_time(pointer["issue_utc"])
        require(SLOT.fullmatch(cycle) and cycle == when.strftime("%Y%m%dT%HZ") and HASH.fullmatch(digest)
                and pointer.get("product") == product and pointer.get("schema") == 1
                and pointer.get("origin") in ("saved-import", "inference"), "Invalid staged metadata.")
        catalog_file = metadata_file.parent / f"catalog-{digest}.json"
        payload = catalog_file.read_bytes()
        require(hexdigest(payload) == digest and len(payload) <= 131072, "Invalid staged catalog hash or size.")
        catalog = json.loads(payload)
        grid, field, units = model_specs(product)
        require(catalog["product"] == product and catalog["issue_utc"] == pointer["issue_utc"]
                and catalog["leads"] == LEADS and catalog["valid_utc"] ==
                [formatted(when + timedelta(hours=lead)) for lead in LEADS]
                and [model["key"] for model in catalog["models"]] == [PRODUCTS[product], "gfs"],
                "Invalid staged catalog identity.")
        files = {catalog_file}
        if pointer["origin"] == "inference":
            pin = metadata_file.parent / "input-receipt.sha256"
            require(HASH.fullmatch(pin.read_text().strip()), "Missing staged provenance hash.")
            files.add(pin)
        for model in catalog["models"]:
            require(model["grid"] == grid and model["field"] == field and model["units"] == units
                    and model["issue_utc"] == pointer["issue_utc"] and len(model["frames"]) == 20,
                    "Invalid staged field.")
            for frame in model["frames"]:
                sha = frame["gzip_sha256"]
                require(HASH.fullmatch(sha) and frame["url"] == f"{ROOT}/{product}/{cycle}/{sha}.bin.gz",
                        "Invalid staged packet URL.")
                file = metadata_file.parent / f"{sha}.bin.gz"
                check_packet(frame, grid, file.read_bytes())
                files.add(file)
        uploads.append((product, cycle, when, pointer, files))
    if dry_run or not uploads:
        if not quiet or dry_run:
            print(f"{len(uploads)} complete product issues verified." if uploads else "No completed issues in the spool.")
        return
    before = output_checkout(checkout)
    for product, cycle, when, pointer, files in uploads:
        base = checkout / "research-live/v1" / product
        latest = base / "latest.json"
        if latest.exists():
            current = read(latest)
            if when <= issue_time(current["issue_utc"]):
                continue
            require(pointer["origin"] == "inference", "A saved import cannot replace a published issue.")
        target = base / cycle
        require(not target.exists(), "The immutable issue directory already exists.")
        target.mkdir(parents=True)
        for file in files:
            shutil.copyfile(file, target / file.name)
        latest.write_bytes(json_bytes(pointer))
    if git(checkout, "status", "--porcelain").stdout.strip():
        git(checkout, "add", "--", "research-live/v1")
        git(checkout, "commit", "-m", "Publish verified TrackFormer research issues")
    head = git(checkout, "rev-parse", "HEAD").stdout.strip()
    if head != before:
        git(checkout, "push", "origin", f"HEAD:refs/heads/{BRANCH}")
        print(f"Research data branch updated at {head}.")
    elif not quiet:
        print("Research data branch already has the latest completed issues.")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    staging = sub.add_parser("stage")
    staging.add_argument("--product", choices=PRODUCTS, required=True)
    staging.add_argument("--catalog", type=Path, required=True)
    staging.add_argument("--packet-root", type=Path, required=True)
    staging.add_argument("--spool", type=Path, required=True)
    source = staging.add_mutually_exclusive_group(required=True)
    source.add_argument("--receipt", type=Path)
    source.add_argument("--saved-import", action="store_true")
    delivery = sub.add_parser("publish")
    delivery.add_argument("--spool", type=Path, required=True)
    delivery.add_argument("--checkout", type=Path, required=True)
    delivery.add_argument("--dry-run", action="store_true")
    delivery.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "stage":
            stage(args.product, args.catalog, args.packet_root, args.spool, args.receipt, args.saved_import)
        else:
            publish(args.spool, args.checkout, args.dry_run, args.quiet)
    except (ValueError, KeyError, StopIteration, OSError, subprocess.CalledProcessError) as error:
        if isinstance(error, subprocess.CalledProcessError):
            # Do not include credential-helper arguments or Git authentication output.
            error = f"Git failed while running {error.cmd[error.cmd.index('user.email=') + 1]} (exit {error.returncode})."
        print(error, file=sys.stderr)
        sys.exit(1)
