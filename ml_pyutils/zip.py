from __future__ import annotations

import zipfile
from pathlib import Path


def validate_zip(
    zip_path: Path,
    required_files: list[Path] | None = None,
    required_dirs: list[Path] | None = None,
) -> tuple[set[Path], set[Path]]:
    """Validate the contents of a ZIP file.

    Args:
        zip_path: Path to the ZIP file to validate.

        required_files: List of files that must be present in the ZIP.

        required_dirs: List of directories that must be present in the ZIP.

    Returns:
        bool: True if the ZIP is valid, False otherwise.
        list[Path]: List of missing files.
        list[Path]: List of missing directories.
    """
    required_files = required_files or []
    required_dirs = required_dirs or []
    with zipfile.ZipFile(zip_path, "r") as z:
        contents = set[Path](Path(n) for n in z.namelist())
        missing_files: set[Path] = set()
        missing_dirs: set[Path] = set()

        for f in required_files:
            if f not in contents:
                missing_files.add(f)

        for d in required_dirs:
            dir_str = str(d).rstrip("/") + "/"  # dir -> dir/, dir/ -> dir/
            if not any(str(x).startswith(dir_str) for x in contents):
                missing_dirs.add(d)

        return missing_files, missing_dirs


def extract_subdir_from_zip(
    zip_path: str | Path, subdir: str | Path, output_dir: str | Path, skip_validation: bool = False
) -> int:
    """Extracts a subdirectory from a ZIP archive into a specified target directory.

    verifies that the zip file exists, the subdirectory path is valid, and copies
    all files and folders preserving directory structure.

    Args:
        zip_path: path to the ZIP file
        subdir: subdirectory inside the ZIP to extract
        output_dir: target folder on disk where the subdirectory will be extracted
    """

    zip_path = Path(zip_path)
    subdir = Path(subdir)
    output_dir = Path(output_dir)

    assert zip_path.is_file(), f"ZIP file does not exist: {zip_path}"
    assert str(subdir).strip(), "subdir path must be non-empty"
    assert str(output_dir).strip(), "output_dir path must be non-empty"

    if skip_validation:
        validate_zip(zip_path=zip_path, required_dirs=[subdir])

    create_counter = 0

    with zipfile.ZipFile(file=zip_path, mode="r") as z:
        members = z.namelist()

        assert members, f"ZIP archive is empty: {zip_path}"

        for member in members:
            member_path = Path(member)

            # skip files not in the target subdir
            # subdir == member_path => subdirectory itself
            # subdir in member_path.parents => subdirectory's children

            if not (subdir == member_path or subdir in member_path.parents):
                continue

            target_path = output_dir / member_path.relative_to(subdir)

            if member.endswith("/"):
                # create directory if it doesn't exist
                target_path.mkdir(parents=True, exist_ok=True)
                create_counter += 1
            else:
                # ensure parent directories exist
                target_path.parent.mkdir(parents=True, exist_ok=True)
                # extract file
                with z.open(name=member) as source, open(file=target_path, mode="wb") as target:
                    target.write(source.read())
                create_counter += 1

    return create_counter


def extract_zip_to_subdirectory(zip_path: str | Path, target_dir: str | Path) -> None:
    """
    Extracts a ZIP archive into a specified subdirectory.

    Ensures that the target directory exists before extraction.
    Raises exceptions if the zip file is invalid or extraction fails.

    Args:
        zip_path: Path to the zip file to extract.
        target_dir: Path to the directory where contents will be extracted.
    """
    # assert inputs are valid
    assert zip_path, "zip_path must be a non-empty string"
    assert target_dir, "target_dir must be a non-empty string"

    zip_file = Path(zip_path)
    extraction_path = Path(target_dir)

    # assert zip file exists
    assert zip_file.is_file(), f"Zip file does not exist: {zip_file}"

    # ensure target directory exists
    extraction_path.mkdir(parents=True, exist_ok=True)

    # extract zip
    try:
        with zipfile.ZipFile(file=zip_file, mode="r") as zip_ref:
            # extract all members to the target directory
            zip_ref.extractall(path=extraction_path)
    except zipfile.BadZipFile as e:
        raise RuntimeError(f"Failed to extract zip file: {zip_file}") from e
