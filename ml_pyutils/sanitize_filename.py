import re
import unicodedata
from pathlib import Path

WINDOWS_RESERVED_NAMES = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        "COM1",
        "COM2",
        "COM3",
        "COM4",
        "COM5",
        "COM6",
        "COM7",
        "COM8",
        "COM9",
        "LPT1",
        "LPT2",
        "LPT3",
        "LPT4",
        "LPT5",
        "LPT6",
        "LPT7",
        "LPT8",
        "LPT9",
    }
)


def sanitize_filename(
    filename: str,
    fallback_filename: str | None = None,
    max_ext_parts: int = 3,
    max_filename_length: int = 255,
) -> str:
    """Sanitize filename for cross-platform compatibility.

    Strategy: Truncate name first, then extension only if name becomes empty.
    """
    if fallback_filename is None:
        fallback_filename = "unnamed.txt"

    if not filename or not isinstance(filename, str):
        return fallback_filename

    # normalize unicode: NFC + strip combining marks
    nfd = unicodedata.normalize("NFD", filename.strip())
    filename = unicodedata.normalize(
        "NFC", "".join(c for c in nfd if unicodedata.category(c) != "Mn")
    )

    # remove nulls, normalize whitespace, remove invalid chars
    filename = filename.replace("\x00", "").replace("\x7f", "")
    filename = re.sub(r"[\xa0\u2003\u3000]", " ", filename)
    filename = re.sub(r'[<>:"/\\|?*\x00-\x1f;`&]', "", filename)

    # collapse dots/spaces, strip trailing
    filename = re.sub(r"\.+", ".", filename)
    filename = re.sub(r" +", " ", filename).rstrip(". ")

    # handle hidden files
    is_hidden = filename.startswith(".")
    filename = (filename[1:] if is_hidden else filename.lstrip(".")).rstrip(".")

    if not filename:
        return f".{fallback_filename}" if is_hidden else fallback_filename

    # split name and extension
    path = Path(filename)
    ext_part = "".join(path.suffixes[-max_ext_parts:])
    name_part = (filename[: -len(ext_part)] if ext_part else filename).rstrip(".")

    if not name_part:
        return f".{fallback_filename}" if is_hidden else fallback_filename

    if is_hidden:
        name_part = f".{name_part}"

    # handle Windows reserved names
    if not is_hidden and name_part.strip().split(".")[0].upper() in WINDOWS_RESERVED_NAMES:
        parts = name_part.split(".", 1)
        name_part = f"{parts[0].strip()}_" + (f".{parts[1]}" if len(parts) > 1 else "")

    # validate extension
    if ext_part:
        ext_part = ext_part if ext_part.startswith(".") else f".{ext_part}"
        if ext_part == ".":
            ext_part = ""

    # truncate to 255 bytes: name first, then extension if name would be empty
    name_bytes = name_part.encode("utf-8")
    ext_bytes = ext_part.encode("utf-8")
    total_bytes = len(name_bytes) + len(ext_bytes)

    if total_bytes > max_filename_length:
        # calculate space for name after reserving extension
        available_for_name = max_filename_length - len(ext_bytes)

        if available_for_name > 0:
            # truncate name, keep full extension
            name_part = name_bytes[:available_for_name].decode("utf-8", errors="ignore")
        else:
            # extension alone exceeds limit, truncate extension and keep minimal name
            min_name_len = min(len(name_bytes), 1)
            name_part = name_bytes[:min_name_len].decode("utf-8", errors="ignore")
            available_for_ext = max_filename_length - len(name_part.encode("utf-8"))
            ext_part = (
                ext_bytes[:available_for_ext].decode("utf-8", errors="ignore")
                if available_for_ext > 0
                else ""
            )

    if not name_part or name_part == ".":
        return f".{fallback_filename}" if is_hidden else fallback_filename

    sanitized = f"{name_part}{ext_part}"
    return sanitized if sanitized and sanitized not in (".", "..") else fallback_filename


if __name__ == "__main__":
    SANITIZE_FILENAME_TEST_CASES = [
        # ============================================================================
        # BASIC VALID FILENAMES
        # ============================================================================
        ("simple.txt", "simple.txt"),
        ("document.pdf", "document.pdf"),
        ("image.png", "image.png"),
        ("video.mp4", "video.mp4"),
        ("archive.zip", "archive.zip"),
        # ============================================================================
        # MULTI-PART EXTENSIONS
        # ============================================================================
        ("backup.tar.gz", "backup.tar.gz"),
        ("archive.tar.bz2", "archive.tar.bz2"),
        ("data.tar.xz", "data.tar.xz"),
        ("bundle.min.js", "bundle.min.js"),
        ("styles.min.css", "styles.min.css"),
        ("package.json.gz", "package.json.gz"),
        ("database.sql.bz2", "database.sql.bz2"),
        ("file.a.b.c.d.e", "file.a.b.c.d.e"),  # max_ext_parts=2 keeps last 2
        # ============================================================================
        # WINDOWS RESERVED NAMES
        # ============================================================================
        ("CON", "CON_"),
        ("con", "con_"),
        ("PRN", "PRN_"),
        ("AUX", "AUX_"),
        ("NUL", "NUL_"),
        ("COM1", "COM1_"),
        ("COM9", "COM9_"),
        ("LPT1", "LPT1_"),
        ("LPT9", "LPT9_"),
        ("CON.txt", "CON_.txt"),
        ("con.txt", "con_.txt"),
        ("PRN.log", "PRN_.log"),
        ("AUX.data", "AUX_.data"),
        ("NUL.bin", "NUL_.bin"),
        ("COM1.cfg", "COM1_.cfg"),
        ("LPT1.out", "LPT1_.out"),
        ("CON.tar.gz", "CON_.tar.gz"),
        ("AUX.min.js", "AUX_.min.js"),
        # ============================================================================
        # INVALID CHARACTERS (WINDOWS)
        # ============================================================================
        ("file<name.txt", "filename.txt"),
        ("file>name.txt", "filename.txt"),
        ("file:name.txt", "filename.txt"),
        ('file"name.txt', "filename.txt"),
        ("file/name.txt", "filename.txt"),
        ("file\\name.txt", "filename.txt"),
        ("file|name.txt", "filename.txt"),
        ("file?name.txt", "filename.txt"),
        ("file*name.txt", "filename.txt"),
        ('file<>:"/\\|?*.txt', "file.txt"),
        # ============================================================================
        # CONTROL CHARACTERS
        # ============================================================================
        ("file\x00name.txt", "filename.txt"),  # null byte
        ("file\x01name.txt", "filename.txt"),  # SOH
        ("file\x1fname.txt", "filename.txt"),  # US
        ("file\x7fname.txt", "filename.txt"),  # DEL
        ("file\nname.txt", "filename.txt"),  # newline
        ("file\rname.txt", "filename.txt"),  # carriage return
        ("file\tname.txt", "filename.txt"),  # tab
        # ============================================================================
        # SHELL-UNSAFE CHARACTERS
        # ============================================================================
        ("file;name.txt", "filename.txt"),  # semicolon
        ("file`name.txt", "filename.txt"),  # backtick
        ("file;rm -rf /.txt", "filerm -rf .txt"),
        # ============================================================================
        # PATH TRAVERSAL ATTEMPTS
        # ============================================================================
        ("../../etc/passwd", ".etcpasswd"),
        ("../../../root/.ssh/id_rsa", ".root.sshid_rsa"),
        (".././.././etc/shadow", ".etcshadow"),
        ("....//....//etc/hosts", ".etchosts"),
        ("/etc/passwd", "etcpasswd"),
        ("C:\\Windows\\System32\\config\\SAM", "CWindowsSystem32configSAM"),
        # ============================================================================
        # DOTS AND SPACES
        # ============================================================================
        ("...file.txt", ".file.txt"),  # leading dots
        ("file....txt", "file.txt"),  # multiple consecutive dots
        ("file....", "file"),  # trailing dots
        ("file  name.txt", "file name.txt"),  # multiple spaces collapsed
        ("  file  .txt", "file .txt"),  # leading/trailing spaces
        ("file.", "file"),  # trailing dot
        ("file..", "file"),  # trailing dots
        ("file...", "file"),  # trailing dots
        (".", "unnamed.txt"),  # just dot
        ("..", "unnamed.txt"),  # parent directory
        ("...", "unnamed.txt"),  # multiple dots
        # ============================================================================
        # UNICODE AND EMOJI
        # ============================================================================
        ("emoji😀.png", "emoji😀.png"),
        ("文件.txt", "文件.txt"),  # Chinese
        ("файл.txt", "фаил.txt"),  # Cyrillic
        ("αρχείο.txt", "αρχειο.txt"),  # Greek
        ("ファイル.txt", "ファイル.txt"),  # Japanese
        ("파일.txt", "파일.txt"),  # Korean
        ("🔥🚀💯.txt", "🔥🚀💯.txt"),
        ("café.txt", "cafe.txt"),  # combining accents
        ("file\u0301name.txt", "filename.txt"),  # combining acute accent (removed)
        # ============================================================================
        # EMPTY AND INVALID INPUTS
        # ============================================================================
        ("", "unnamed.txt"),
        ("   ", "unnamed.txt"),
        (".txt", ".txt"),  # no name, just extension
        ("..txt", ".txt"),
        ("...txt", ".txt"),
        (".........txt", ".txt"),
        (None, "unnamed.txt"),  # handled by type check
        # ============================================================================
        # LONG FILENAMES (TRUNCATION)
        # ============================================================================
        ("a" * 300 + ".txt", "a" * 251 + ".txt"),  # 251 + 4 = 255 bytes
        ("文" * 100 + ".txt", None),  # each char = 3 bytes UTF-8, gets truncated
        ("😀" * 100 + ".txt", None),  # each emoji = 4 bytes, gets truncated
        ("filename_" + "x" * 250 + ".tar.gz", None),  # long name with multi-part ext
        # ============================================================================
        # EXTENSION EDGE CASES
        # ============================================================================
        ("file.e̊xe", "file.exe"),  # combining diacritics removed
        ("file.txt.", "file.txt"),  # trailing dot after extension
        ("file..txt", "file.txt"),  # double dot before extension
        ("file..", "file"),  # just dots as extension
        ("file.", "file"),  # empty extension
        ("file.a.b.c.d.e.f.g", "file.a.b.c.d.e.f.g"),  # many extensions
        ("file" + ".x" * 50, None),  # extremely long multi-part extension
        # ============================================================================
        # CASE SENSITIVITY
        # ============================================================================
        ("FILE.TXT", "FILE.TXT"),  # uppercase preserved
        ("File.Txt", "File.Txt"),  # mixed case preserved
        ("con.TXT", "con_.TXT"),  # reserved name check is case-insensitive
        ("CON.txt", "CON_.txt"),
        # ============================================================================
        # SPECIAL COMBINATIONS
        # ============================================================================
        ("con.txt.exe", "con_.txt.exe"),  # reserved name with double extension
        ("file name with spaces.tar.gz", "file name with spaces.tar.gz"),
        ("my-file_v2.final.backup.tar.gz", "my-file_v2.final.backup.tar.gz"),
        ("../../../etc/passwd.tar.gz", ".etcpasswd.tar.gz"),
        ("CON...txt", "CON_.txt"),  # reserved + multiple dots
        ("  CON  .txt  ", "CON_.txt"),  # reserved + spaces
        # ============================================================================
        # REALISTIC FILENAMES
        # ============================================================================
        ("Project Report (Final).docx", "Project Report (Final).docx"),
        ("2024-01-15_Meeting_Notes.pdf", "2024-01-15_Meeting_Notes.pdf"),
        ("IMG_20240115_143022.jpg", "IMG_20240115_143022.jpg"),
        ("Screen Shot 2024-01-15 at 2.30.45 PM.png", "Screen Shot 2024-01-15 at 2.30.45 PM.png"),
        ("database_backup_2024-01-15.sql.gz", "database_backup_2024-01-15.sql.gz"),
        ("client-project-v2.3-final-FINAL.zip", "client-project-v2.3-final-FINAL.zip"),
        # ============================================================================
        # EDGE CASES WITH COMBINING MARKS
        # ============================================================================
        ("café\u0301.txt", "cafe.txt"),  # extra combining mark removed
        ("file\u0300\u0301\u0302.txt", "file.txt"),  # multiple combining marks
        # ============================================================================
        # WHITESPACE VARIATIONS
        # ============================================================================
        ("file\u00a0name.txt", "file name.txt"),  # non-breaking space (preserved)
        ("file\u2003name.txt", "file name.txt"),  # em space (preserved)
        ("file\u3000name.txt", "file name.txt"),  # ideographic space (preserved)
        # ============================================================================
        # EXTREMELY LONG NAMES WITH UNICODE
        # ============================================================================
        ("日本語" * 100 + ".txt", None),  # long Japanese text
        ("🚀" * 70 + ".tar.gz", None),  # long emoji string
        # ============================================================================
        # INJECTION ATTEMPTS
        # ============================================================================
        ("file; rm -rf /.txt", "file rm -rf .txt"),
        ("file`cat /etc/passwd`.txt", "filecat etcpasswd.txt"),
        ("file && echo hacked.txt", "file echo hacked.txt"),
        ("file | tee /dev/null.txt", "file tee devnull.txt"),
        # ============================================================================
        # BOUNDARY CONDITIONS
        # ============================================================================
        ("a.txt", "a.txt"),  # single char name
        ("ab.txt", "ab.txt"),  # two char name
        ("a" * 251 + ".txt", "a" * 251 + ".txt"),  # exactly 255 bytes
        ("a" * 252 + ".txt", "a" * 251 + ".txt"),  # 256 bytes, truncated to 255
        # ============================================================================
        # WINDOWS RESERVED NAMES IN DIFFERENT CONTEXTS
        # ============================================================================
        ("file_CON.txt", "file_CON.txt"),  # CON not at start
        ("CON_file.txt", "CON_file.txt"),  # CON at start but not alone
        ("myCON.txt", "myCON.txt"),  # CON as part of word
        ("CONfig.txt", "CONfig.txt"),  # CON prefix but not reserved
        # ============================================================================
        # EMPTY EXTENSIONS
        # ============================================================================
        ("file..", "file"),  # double dot
        ("file...", "file"),  # triple dot
        ("file....", "file"),  # quad dot
        # ============================================================================
        # empty filenames
        # ============================================================================
        ("././", "unnamed.txt"),
        (".", "unnamed.txt"),
        ("....", "unnamed.txt"),
        ("//", "unnamed.txt"),
        ("/././", "unnamed.txt"),
        ("../", "unnamed.txt"),
        ("./.", "unnamed.txt"),
        ("....//..././././..", "unnamed.txt"),
        ("ab." + "x" * 255, "a." + "x" * 253),  # total length 255 bytes
        ("ab." + "x" * 257, "a." + "x" * 253),  # total length 255 bytes
    ]

    LONG_FILENAME_CASES = [("x" * n + ".txt", None) for n in range(250, 260)]

    UNICODE_LENGTH_CASES = [
        # Multi-byte UTF-8 characters
        ("文" * n + ".txt", None)
        for n in range(80, 90)  # 3 bytes each
    ]

    EMOJI_LENGTH_CASES = [
        # 4-byte emoji characters
        ("😀" * n + ".txt", None)
        for n in range(60, 70)  # 4 bytes each
    ]

    ALL_TEST_CASES = (
        SANITIZE_FILENAME_TEST_CASES
        + LONG_FILENAME_CASES
        + UNICODE_LENGTH_CASES
        + EMOJI_LENGTH_CASES
    )

    for input_name, expected in SANITIZE_FILENAME_TEST_CASES:
        result = sanitize_filename(input_name)
        if expected is not None and result != expected:
            print(f"Failed: {input_name!r} -> {result!r} (expected {expected!r})")
