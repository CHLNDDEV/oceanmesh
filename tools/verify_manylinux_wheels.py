from __future__ import annotations

import argparse
import re
import subprocess
import sys
import zipfile
from pathlib import Path


def _parse_glibc_symbols(auditwheel_verbose_output: str) -> tuple[int, int] | None:
    matches = re.findall(r"GLIBC_(\d+)\.(\d+)", auditwheel_verbose_output)
    if not matches:
        return None
    versions = [(int(major), int(minor)) for major, minor in matches]
    return max(versions)


def _run_auditwheel_show_verbose(wheel_path: Path) -> str:
    return subprocess.check_output(
        [sys.executable, "-m", "auditwheel", "show", "-v", str(wheel_path)],
        text=True,
        stderr=subprocess.STDOUT,
    )


def _wheel_contains_shared_lib(wheel_path: Path, needle: str) -> bool:
    # auditwheel typically vendors libs under something like:
    #   <pkg>.libs/libgmp-....so
    # so we just search for the substring in zip members.
    with zipfile.ZipFile(wheel_path, "r") as zf:
        for name in zf.namelist():
            # Bundled libs are commonly version-suffixed (e.g. .so.10, .so.6).
            if needle in name and ".so" in name:
                return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dist_dir",
        type=Path,
        help="Directory containing built wheels (e.g. dist/)",
    )
    parser.add_argument(
        "--max-glibc",
        default="2.28",
        help="Maximum allowed GLIBC symbol version (default: 2.28)",
    )
    args = parser.parse_args()

    max_glibc_parts = args.max_glibc.split(".")
    if len(max_glibc_parts) != 2:
        raise SystemExit("--max-glibc must look like '2.28'")

    max_allowed = (int(max_glibc_parts[0]), int(max_glibc_parts[1]))

    wheels = sorted(args.dist_dir.glob("*.whl"))
    linux_wheels = [
        w
        for w in wheels
        if "linux" in w.name and "x86_64" in w.name and "musllinux" not in w.name
    ]

    if not linux_wheels:
        raise SystemExit(f"No Linux x86_64 wheels found in {args.dist_dir}")

    failed = False

    for wheel in linux_wheels:
        print(f"\n=== Checking wheel: {wheel.name} ===")

        if "manylinux" not in wheel.name:
            print("ERROR: wheel filename does not include a manylinux tag")
            failed = True

        if "linux_x86_64" in wheel.name and "manylinux" not in wheel.name:
            print("ERROR: wheel appears to be plain linux_x86_64 (not manylinux)")
            failed = True

        out = _run_auditwheel_show_verbose(wheel)
        print(out)

        max_seen = _parse_glibc_symbols(out)
        if max_seen is not None and max_seen > max_allowed:
            print(
                f"ERROR: wheel references too-new GLIBC symbols: "
                f"max seen GLIBC_{max_seen[0]}.{max_seen[1]} > allowed GLIBC_{max_allowed[0]}.{max_allowed[1]}"
            )
            failed = True

        # Ensure auditwheel actually bundled GMP/MPFR into the wheel
        has_gmp = _wheel_contains_shared_lib(wheel, "libgmp")
        has_mpfr = _wheel_contains_shared_lib(wheel, "libmpfr")

        if not has_gmp:
            print("ERROR: bundled libgmp*.so not found inside wheel")
            failed = True
        if not has_mpfr:
            print("ERROR: bundled libmpfr*.so not found inside wheel")
            failed = True

        if has_gmp and has_mpfr:
            print("OK: wheel contains bundled libgmp + libmpfr")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
