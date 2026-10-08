"""Exact native CMORPH transport only; no model, labels, OAuth or training.

The only Drive operations are PUTs to pre-authorized, single-file upload
sessions. The runner cannot enumerate, create, move or delete Drive files.
Native scientific decoding and corpus/trainer qualification remain pending.
"""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

PLAN_ID = "8a652626b3406d43ec44786d43f1add57f04ec83ac414f2c2805b04308aad868"
PROJECT = "1o0nqk6Taxd-HoGcTDSe_9TomRn4ZV3fj"
PRODUCT = "NOAA_CMORPH_CDR_V1.0_ADJ_8KM_30MIN"
SOURCE_HOST = "noaa-cdr-precip-cmorph-pds.s3.amazonaws.com"
ARCHIVE = "https://" + SOURCE_HOST + "/data/30min/8km"
MAX_SOURCE = 64 * 1024**2
MAX_PART = 4 * 1024**3
CHUNK = 8 * 1024**2
CAP_KEYS = ["V1312_UPLOAD_CAPABILITIES_" + str(i) for i in range(8)]


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def identity(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def check_plan(plan):
    value = dict(plan)
    expected = value.pop("identity")
    ends = plan["rain_interval_end_seconds"]
    if (identity(value) != expected or plan["rain_plan_identity"] != PLAN_ID or
            plan["project_folder_id"] != PROJECT or plan["product"] != PRODUCT or
            plan["counts"] != {"train": 11978, "development": 811} or
            len(ends) != 13613 or ends != sorted(set(ends)) or
            any(type(t) is not int or t % 21600 for t in ends) or
            plan["batch_hours"] != 1024 or plan["initial_batch_hours"] != 64 or plan["workers"] != 32 or
            plan["training_started"] or plan["scientific_corpus_ready"]):
        raise ValueError("Frozen transport plan failed")
    hours = sorted({end - 21600 + i * 3600 for end in ends for i in range(6)})
    if len(hours) != 81678 or identity(hours) != plan["hours_identity"]:
        raise ValueError("Exact source-hour coverage failed")
    return hours


def parts(hours, plan):
    first = plan["initial_batch_hours"]
    yield hours[:first]
    for start in range(first, len(hours), plan["batch_hours"]):
        yield hours[start:start + plan["batch_hours"]]


def source_location(hour):
    d = datetime.fromtimestamp(hour, timezone.utc)
    relative = f"native/{d:%Y/%m/%d}/CMORPH_V1.0_ADJ_8km-30min_{d:%Y%m%d%H}.nc"
    return ARCHIVE + "/" + relative.removeprefix("native/"), relative


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("Redirect rejected by transport allowlist")


OPENER = urllib.request.build_opener(NoRedirect)


def check_session(uri):
    p = urllib.parse.urlsplit(uri)
    query = urllib.parse.parse_qs(p.query, strict_parsing=True)
    if (p.scheme != "https" or p.hostname != "www.googleapis.com" or p.port is not None or
            p.username is not None or p.password is not None or p.fragment or
            p.path != "/upload/drive/v3/files" or
            query.get("uploadType") != ["resumable"] or
            len(query.get("upload_id", [])) != 1 or not query["upload_id"][0]):
        raise ValueError("Single-file upload capability rejected")
    return uri


def load_caps(plan, environ=os.environ):
    entries = []
    for key in CAP_KEYS:
        value = environ.pop(key, "")
        if not value:
            raise ValueError("Upload capability group missing")
        decoded = gzip.decompress(base64.b64decode(value, validate=True))
        if len(decoded) > 256 * 1024:
            raise ValueError("Oversized capability group")
        group = json.loads(decoded)
        if group["transport_identity"] != plan["identity"]:
            raise ValueError("Capability/transport identity mismatch")
        entries.extend(group["entries"])
    expected = 1 + (plan["native_hour_count"] - plan["initial_batch_hours"] + plan["batch_hours"] - 1) // plan["batch_hours"]
    names = [part_name(i) for i in range(expected)] + ["complete-native-source-receipt.json"]
    if [e["name"] for e in entries] != names or len({e["session"] for e in entries}) != len(entries):
        raise ValueError("Missing, duplicate or replaced upload destinations")
    for e in entries:
        if e["parent_id"] != plan["destination_folder_id"]:
            raise ValueError("Upload destination outside the frozen folder")
        check_session(e["session"])
        # Do not rely on GitHub masking the compressed secret's derived values.
        if os.environ.get("GITHUB_ACTIONS") == "true":
            print("::add-mask::" + e["session"], flush=True)
            print("::add-mask::" + urllib.parse.parse_qs(urllib.parse.urlsplit(e["session"]).query)["upload_id"][0], flush=True)
    return entries


def part_name(i):
    return f"cmorph-native-{i:04d}.tar"


def ensure_disk(directory, additional=0):
    if shutil.disk_usage(directory).free < 2 * 1024**3 + additional:
        raise RuntimeError("Cloud scratch reserve reached; no persistent data removed")


def fetch_hour(hour, directory):
    url, relative = source_location(hour)
    path = Path(directory) / Path(relative).name
    for attempt in range(3):
        try:
            ensure_disk(directory, MAX_SOURCE)
            h, size = hashlib.sha256(), 0
            req = urllib.request.Request(url, headers={"User-Agent": "Trackformer-1.3.12-exact-native-transport"})
            with OPENER.open(req, timeout=90) as response, path.open("wb") as out:
                if response.status != 200:
                    raise ValueError("Unexpected native source response")
                length = int(response.headers.get("Content-Length", 0))
                for block in iter(lambda: response.read(1024**2), b""):
                    size += len(block)
                    if size > MAX_SOURCE:
                        raise ValueError("Native source exceeds unchanged 64-MiB bound")
                    out.write(block)
                    h.update(block)
                out.flush()
                os.fsync(out.fileno())
            if size <= 0 or (length and size != length):
                raise ValueError("Truncated source")
            with path.open("rb") as f:
                magic = f.read(8)
            if magic != b"\x89HDF\r\n\x1a\n" and magic[:3] != b"CDF":
                raise ValueError("Native netCDF container signature failed")
            return path, dict(path=relative, hour_seconds=hour, url=url, size=size, sha256=h.hexdigest(),
                primary_labels=PRODUCT, role="retrospective_training_label_only",
                scientific_geometry_interval_mask_audit_pending=True)
        except (urllib.error.URLError, TimeoutError, OSError):
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("Bounded source retries exhausted")


def tar_bytes(tar, name, data):
    info = tarfile.TarInfo(name)
    info.size, info.mtime, info.mode = len(data), 0, 0o644
    tar.addfile(info, io.BytesIO(data))


def make_part(hours, target, pool, directory, plan, index):
    records = []
    with tarfile.open(target, "w", format=tarfile.USTAR_FORMAT) as archive:
        for start in range(0, len(hours), plan["workers"]):
            window = hours[start:start + plan["workers"]]
            futures = [pool.submit(fetch_hour, h, directory) for h in window]
            for future in futures:
                path, record = future.result()
                if Path(target).stat().st_size + record["size"] > MAX_PART:
                    raise RuntimeError("Cloud part size bound reached")
                ensure_disk(directory, record["size"])
                info = tarfile.TarInfo(record["path"])
                info.size, info.mtime, info.mode = record["size"], 0, 0o644
                with path.open("rb") as f:
                    archive.addfile(info, f)
                tar_bytes(archive, record["path"] + ".receipt.json", canonical(record))
                records.append(record)
                # Only our ephemeral downloaded scratch copy; persistent archives
                # and all Mac/checkpoint data are untouched.
                path.unlink()
            print(f"NATIVE_PROGRESS part={index} exact_hours={len(records)}/{len(hours)}", flush=True)
        tar_bytes(archive, "cloud-part-manifest.json", canonical(dict(
            schema="v1312-native-source-part-v1", transport_identity=plan["identity"],
            rain_plan_identity=PLAN_ID, part=index, sources=records,
            native_bytes_unchanged=True, scientific_corpus_ready=False, training_started=False)))
    with Path(target).open("rb") as f:
        os.fsync(f.fileno())
    return records


def upload_status(uri, total):
    req = urllib.request.Request(check_session(uri), data=b"", method="PUT",
        headers={"Content-Length": "0", "Content-Range": f"bytes */{total}"})
    try:
        with OPENER.open(req, timeout=90) as response:
            return response.status, {k.lower(): v for k, v in response.headers.items()}, response.read()
    except urllib.error.HTTPError as e:
        if e.code == 308:
            result = e.code, {k.lower(): v for k, v in e.headers.items()}, e.read()
            e.close()
            return result
        raise


def upload(path, cap):
    uri, total = check_session(cap["session"]), Path(path).stat().st_size
    offset, failures, metadata = 0, 0, None
    with Path(path).open("rb") as f:
        while offset < total:
            f.seek(offset)
            data = f.read(min(CHUNK, total - offset))
            req = urllib.request.Request(uri, data=data, method="PUT", headers={
                "Content-Type": "application/octet-stream", "Content-Length": str(len(data)),
                "Content-Range": f"bytes {offset}-{offset + len(data)-1}/{total}"})
            try:
                try:
                    with OPENER.open(req, timeout=180) as r:
                        code, headers, body = r.status, {k.lower(): v for k, v in r.headers.items()}, r.read()
                except urllib.error.HTTPError as e:
                    if e.code != 308:
                        raise
                    code, headers, body = e.code, {k.lower(): v for k, v in e.headers.items()}, e.read()
                    e.close()
                if code in (200, 201):
                    metadata = json.loads(body)
                    break
                if code != 308:
                    raise ValueError("Unexpected upload response")
                match = re.fullmatch(r"bytes=0-(\d+)", headers.get("range", ""))
                new_offset = int(match[1]) + 1 if match else 0
                if new_offset <= offset or new_offset > total:
                    raise ValueError("Upload made no verified forward progress")
                offset = new_offset
            except (urllib.error.URLError, TimeoutError, OSError):
                failures += 1
                if failures > 5:
                    raise RuntimeError("Bounded Drive upload retries exhausted") from None
                time.sleep(min(2 ** failures, 30))
                code, headers, body = upload_status(uri, total)
                if code in (200, 201):
                    metadata = json.loads(body)
                    break
                match = re.fullmatch(r"bytes=0-(\d+)", headers.get("range", ""))
                offset = int(match[1]) + 1 if match else 0
                if not 0 <= offset <= total:
                    raise ValueError("Invalid resumed byte range")
    if (not metadata or metadata.get("name") != cap["name"] or
            metadata.get("parents") != [cap["parent_id"]] or
            int(metadata.get("size", -1)) != total or
            metadata.get("md5Checksum") != file_hash(path, "md5")):
        raise ValueError("Server destination/size/checksum verification failed")
    if metadata.get("sha256Checksum") and metadata["sha256Checksum"] != file_hash(path):
        raise ValueError("Server SHA256 verification failed")
    return dict(file_id=metadata["id"], name=cap["name"], parent_id=cap["parent_id"],
        size=total, sha256=file_hash(path), server_md5=metadata["md5Checksum"],
        server_sha256=metadata.get("sha256Checksum"), server_checksum_verified=True)


def run(plan, caps, directory):
    hours = check_plan(plan)
    Path(directory).mkdir(parents=True, exist_ok=True)
    ensure_disk(directory)
    print("CAPABILITIES_ACCEPTED: single-file destinations only; no Drive OAuth token", flush=True)
    parts, completed = [], 0
    with ThreadPoolExecutor(max_workers=plan["workers"]) as pool:
        for i, selected in enumerate(parts(hours, plan)):
            with tempfile.TemporaryDirectory(prefix="cmorph-part-", dir=directory) as work:
                path = Path(work) / part_name(i)
                records = make_part(selected, path, pool, work, plan, i)
                remote = upload(path, caps[i])
                remote.update(source_count=len(records), first_hour=selected[0], last_hour=selected[-1])
                parts.append(remote)
                completed += len(records)
                print("VERIFIED_PART " + json.dumps(dict(part=i, completed_native_hours=completed,
                    total_native_hours=len(hours), **remote), sort_keys=True), flush=True)
    if completed != 81678 or sum(p["source_count"] for p in parts) != len(hours):
        raise ValueError("Full exact native coverage not completed")
    receipt = dict(schema="v1312-full-native-cloud-download-v1", transport_identity=plan["identity"],
        rain_plan_identity=PLAN_ID, native_hour_count=completed, parts=parts,
        all_sources_full_sha256_in_part_manifests=True, cloud_download_complete=True,
        scientific_decode_interval_area_masks_pending=True, scientific_corpus_ready=False,
        full_training_bundle_ready=False, training_started=False,
        completed_utc=datetime.now(timezone.utc).isoformat())
    with tempfile.TemporaryDirectory(prefix="cmorph-final-", dir=directory) as work:
        path = Path(work) / caps[-1]["name"]
        path.write_bytes(canonical(receipt))
        final = upload(path, caps[-1])
    print("COMPLETE_NATIVE_DOWNLOAD " + json.dumps(final, sort_keys=True), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", required=True, type=Path)
    p.add_argument("--scratch", required=True, type=Path)
    args = p.parse_args()
    plan = json.loads(args.plan.read_text())
    check_plan(plan)
    caps = load_caps(plan)
    run(plan, caps, args.scratch)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # Never print request objects, exception URLs, capabilities or tracebacks.
        print("CLOUD_DOWNLOAD_STOPPED_SAFE: " + type(exc).__name__ + "; no training started", flush=True)
        raise SystemExit(1)
