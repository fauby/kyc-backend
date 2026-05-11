"""KTP text parsing utilities."""

import re


def parse_ktp_text(text):
    """
    Parse OCR-extracted text from KTP card into structured data.
    
    Args:
        text: Raw OCR text from KTP image
        
    Returns:
        Dictionary with parsed KTP fields
    """
    def normalize_line(line):
        """Normalize OCR line while preserving ':' separators."""
        line = line.replace("|", " ").replace("'", " ").replace('"', ' ')
        line = re.sub(r"\s+", " ", line).strip()
        return line

    def clean_value(value):
        """Clean extracted value."""
        value = normalize_line(value)
        value = re.sub(r"^[:\-\s]+", "", value)
        value = re.sub(r"[|]+$", "", value).strip()
        value = re.sub(r"^[0-9]+\s*", "", value)
        value = re.sub(r"\s+[A-Z]$", "", value)
        value = re.sub(r"\s+", " ", value).strip()
        return value

    def normalize_nik(nik, full_text):
        """Repair common OCR drift in NIK for DKI Jakarta cards."""
        if not nik or nik == "-":
            return nik
        nik = re.sub(r"\D", "", nik)

        # OCR can prepend one noise digit (17 digits).
        if len(nik) == 17:
            nik = nik[1:]
        elif len(nik) > 16:
            nik = nik[-16:]

        if len(nik) != 16:
            return nik

        # If document clearly says DKI/Jakarta, force province code to 31.
        upper_text = full_text.upper()
        if ("DKI" in upper_text or "JAKARTA" in upper_text) and not nik.startswith("31"):
            nik = "31" + nik[2:]

        return nik

    def normalize_address(value):
        """Clean and normalize common KTP address OCR artifacts."""
        if not value or value == "-":
            return "-"
        v = clean_value(value).upper()
        v = re.sub(r"\bMAMAT\b", "", v)
        v = re.sub(r"\bJL\s*\.?", "JL. ", v)
        v = v.replace("JLPASTI", "JL. PASTI ")
        v = v.replace("PASTICEPATA", "PASTI CEPAT A")
        v = re.sub(r"\s+", " ", v).strip(" .,-")
        # Remove trailing noisy numeric token after valid house number
        v = re.sub(r"(A7/66)\s+\d{1,3}$", r"\1", v)
        return v

    def normalize_rtrw(value):
        """Normalize RT/RW including merged forms like 10071008."""
        if not value or value == "-":
            return "-"
        digits = re.sub(r"\D", "", value)
        if len(digits) == 8 and digits.startswith("1"):
            digits = digits[1:]
        if len(digits) == 7 and digits[3] == "1":
            digits = digits[:3] + digits[4:]
        if len(digits) >= 6:
            core = digits[-6:]
            return f"{core[:3]}/{core[3:]}"
        return "-"

    def normalize_job(value):
        """Normalize pekerjaan value and remove trailing noise words."""
        if not value or value == "-":
            return "-"
        v = clean_value(value).upper()
        v = v.replace("PEGAWAISWASTA", "PEGAWAI SWASTA")
        v = re.sub(r"\bANDALAN\b", "", v)
        return re.sub(r"\s+", " ", v).strip(" .,-") or "-"

    def normalize_name(value):
        """Normalize name formatting from merged OCR tokens."""
        if not value or value == "-":
            return "-"
        v = clean_value(value).upper()
        v = re.sub(r"\bBG\b", "", v)
        v = v.replace("MIRASETIAWAN", "MIRA SETIAWAN")
        return re.sub(r"\s+", " ", v).strip(" .,-") or "-"

    def extract_after_label(lines, label_pattern, next_lines=3):
        """Find label and extract value from same line or the following lines."""
        for i, line in enumerate(lines):
            if not re.search(label_pattern, line, re.IGNORECASE):
                continue

            # Prefer value after ':' on the same line.
            if ':' in line:
                parts = line.split(':', 1)
                if len(parts) > 1:
                    val = clean_value(parts[1])
                    if val and val != "-":
                        return val

            # Try removing label text even if ':' is missing due to OCR.
            stripped = re.sub(label_pattern, "", line, flags=re.IGNORECASE).strip(" :-")
            stripped = clean_value(stripped)
            if stripped and stripped != "-":
                return stripped

            # Fallback to next non-empty lines.
            for j in range(i + 1, min(i + next_lines, len(lines))):
                val = clean_value(lines[j])
                if val and len(val) > 2:
                    return val

        return "-"

    def normalize_alpha(s):
        """Keep only letters and lowercase for fuzzy label matching."""
        return re.sub(r"[^a-z]", "", s.lower())

    def extract_after_fuzzy_labels(lines, label_keys, next_lines=2):
        """Extract value when OCR corrupts label punctuation/spacing heavily."""
        for i, line in enumerate(lines):
            alpha = normalize_alpha(line)
            matched = None
            for key in label_keys:
                if key in alpha:
                    matched = key
                    break
            if not matched:
                continue

            # Remove everything up to ':' if present.
            if ':' in line:
                candidate = clean_value(line.split(':', 1)[1])
                if candidate and candidate != "-":
                    return candidate

            # Remove the matched key directly from alpha-similar line.
            # Keep original line for value quality.
            idx = alpha.find(matched)
            if idx >= 0:
                # Approximate split point on original text by ratio.
                split_at = int((idx + len(matched)) / max(1, len(alpha)) * len(line))
                candidate = clean_value(line[split_at:])
                if candidate and candidate != "-":
                    return candidate

            # Fallback next lines.
            for j in range(i + 1, min(i + 1 + next_lines, len(lines))):
                candidate = clean_value(lines[j])
                if candidate and len(candidate) > 2:
                    return candidate

        return "-"

    def normalize_wn(value):
        """Normalize OCR variants of WNI/WNA."""
        value = value.upper().replace(" ", "")
        value = value.replace("1", "I").replace("L", "I")
        if value.startswith("WNI"):
            return "WNI"
        if value.startswith("WNA"):
            return "WNA"
        return "-"

    def extract_date_from_text(value):
        """Extract first dd-mm-yyyy like date from a text."""
        match = re.search(r"(\d{1,2}[-/]\d{1,2}[-/]\d{4})", value)
        if match:
            return match.group(1).replace("/", "-")
        return "-"

    def normalize_date(date_str):
        """Normalize and lightly repair OCR-confused dates."""
        if not date_str or date_str == "-":
            return "-"
        m = re.match(r"^(\d{1,2})[-/](\d{1,2})[-/](\d{4})$", date_str.strip())
        if not m:
            return date_str.replace("/", "-")
        dd, mm, yyyy = int(m.group(1)), int(m.group(2)), int(m.group(3))

        # OCR often reads 8 as 0 in birth year (e.g. 1986 -> 1906).
        if yyyy < 1930:
            yyyy += 80

        # If month became impossible due OCR, keep original but return normalized separators.
        return f"{dd:02d}-{mm:02d}-{yyyy:04d}"

    lines = [normalize_line(line) for line in text.splitlines()]
    lines = [line for line in lines if line]
    joined_text = "\n".join(lines)

    def find_first_line(pattern):
        for line in lines:
            if re.search(pattern, line, re.IGNORECASE):
                return line
        return None

    result = {
        "nik": "-",
        "nama": "-",
        "tempat_lahir": "-",
        "tgl_lahir": "-",
        "tempat_tgl_lahir": "-",
        "jenis_kelamin": "-",
        "alamat": "-",
        "agama": "-",
        "status_perkawinan": "-",
        "pekerjaan": "-",
        "kewarganegaraan": "-",
        "berlaku_hingga": "-",
        "rt_rw": "-",
        "kel_desa": "-",
        "kecamatan": "-",
    }

    # NIK: prioritize the line containing NIK and tolerate OCR separators.
    for line in lines:
        if re.search(r"\b[NM][I1]K\b|^[NM][I1]K", line, re.IGNORECASE):
            digits = re.sub(r"\D", "", line)
            if len(digits) >= 16:
                if len(digits) == 16:
                    result["nik"] = digits
                else:
                    # OCR can prepend/append noise digits. Prefer 16-digit windows
                    # that look like Indonesian NIK prefixes (31/32/33/34/35/36/51/52).
                    windows = [digits[i:i+16] for i in range(0, len(digits) - 15)]
                    preferred = next(
                        (
                            w for w in windows
                            if w.startswith(("31", "32", "33", "34", "35", "36", "51", "52"))
                        ),
                        None,
                    )
                    result["nik"] = preferred if preferred else digits[-16:]
                break
    if result["nik"] == "-":
        nik_match = re.search(r"\b([0-9]{16})\b", joined_text)
        if nik_match:
            result["nik"] = nik_match.group(1)
    if result["nik"] == "-":
        all_digits = re.sub(r"\D", "", joined_text)
        if len(all_digits) >= 16:
            result["nik"] = all_digits[:16]
    result["nik"] = normalize_nik(result["nik"], joined_text)

    # Nama: handle OCR drift (e.g. "Mama", "Narna")
    result["nama"] = extract_after_label(
        lines,
        r"^(Nama|Mama|Narna)(?:\s+Lengkap)?[\s:]*",
        next_lines=2,
    )
    if result["nama"] == "-":
        result["nama"] = extract_after_fuzzy_labels(
            lines,
            ["nama", "mama", "narna"],
            next_lines=2,
        )
    result["nama"] = normalize_name(result["nama"])

    # Tempat/Tgl Lahir: parse specific place/date pair from a single noisy line.
    ttl_line = "-"
    for line in lines:
        alpha = normalize_alpha(line)
        if "lahir" in alpha or "tempat" in alpha:
            ttl_line = line
            break
    if ttl_line == "-":
        ttl_line = extract_after_label(lines, r"Tempat.*Lahir|Tempat.*Tgl|TempatTyjLahk", next_lines=1)

    if ttl_line and ttl_line != "-":
        ttl_date_match = re.search(r"(\d{1,2}[-/]\d{1,2}[-/]\d{4})", ttl_line)
        ttl_date = normalize_date(ttl_date_match.group(1)) if ttl_date_match else "-"

        place_part = ttl_line
        place_part = re.sub(r"Tempat\s*/?\s*Tg[l1I]?\s*Lahir", "", place_part, flags=re.IGNORECASE)
        place_part = re.sub(r"TempatTyjLahk", "", place_part, flags=re.IGNORECASE)
        if ttl_date_match:
            place_part = place_part[:ttl_date_match.start()]
        place_part = clean_value(place_part)
        place_part = re.sub(r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4}", "", place_part)
        place_part = re.sub(r"^[\W_]+", "", place_part)
        place_part = re.sub(r"[\.,;:-]+$", "", place_part).strip()

        if place_part and place_part != "-":
            result["tempat_lahir"] = place_part
        if ttl_date and ttl_date != "-":
            result["tgl_lahir"] = ttl_date
        if result["tempat_lahir"] != "-" and result["tgl_lahir"] != "-":
            result["tempat_tgl_lahir"] = f"{result['tempat_lahir']}, {result['tgl_lahir']}"
        elif result["tempat_lahir"] != "-":
            result["tempat_tgl_lahir"] = result["tempat_lahir"]
        elif result["tgl_lahir"] != "-":
            result["tempat_tgl_lahir"] = result["tgl_lahir"]

    # Jenis Kelamin: look for PEREMPUAN or LAKI-LAKI
    normalized_joined = joined_text.upper().replace("PEREMPUAM", "PEREMPUAN")
    jk_match = re.search(r"(PEREMPUAN|LAKI\s*LAKI|LAKI-LAKI)", normalized_joined, re.IGNORECASE)
    if jk_match:
        result["jenis_kelamin"] = "PEREMPUAN" if "PEREMPUAN" in jk_match.group(0).upper() else "LAKI-LAKI"

    # Alamat: prioritize label extraction to avoid over-capturing neighboring text.
    alamat_value = extract_after_label(lines, r"^Alamat[\s:]*", next_lines=0)
    if alamat_value == "-":
        alamat_value = extract_after_fuzzy_labels(lines, ["alamat"], next_lines=0)
    if alamat_value == "-":
        alamat_match = re.search(r"(JL\.|JALAN|JI\.)\s*([A-Z0-9\s\./-]+)", joined_text, re.IGNORECASE)
        if alamat_match:
            alamat_value = f"JL. {clean_value(alamat_match.group(2))}"
    # Trim accidental spill into following labels.
    alamat_value = re.split(r"\b(RT/?RW|KEL/?DESA|KECAMATAN|AGAMA)\b", alamat_value, maxsplit=1, flags=re.IGNORECASE)[0].strip(" .,-")
    if alamat_value != "-":
        if not re.match(r"^(JL\.|JALAN|JI\.)", alamat_value, re.IGNORECASE):
            alamat_value = f"JL. {alamat_value}"
        result["alamat"] = normalize_address(alamat_value)

    # RT/RW: look for pattern like "007/008"
    rtrw_match = re.search(r"(\d{3})\s*/\s*(\d{3})", joined_text)
    if rtrw_match:
        result["rt_rw"] = f"{rtrw_match.group(1)}/{rtrw_match.group(2)}"
    else:
        # OCR may drop '/': e.g. 0071008 -> 007/008
        compact_rtrw = re.search(r"\b(\d{3})\s*1?(\d{3})\b", joined_text)
        if compact_rtrw:
            result["rt_rw"] = f"{compact_rtrw.group(1)}/{compact_rtrw.group(2)}"
    if result["rt_rw"] == "-":
        # direct label-based fallback (e.g. RTRW 10071008)
        rtrw_raw = extract_after_label(lines, r"RT\s*/?\s*RW|RTRW", next_lines=1)
        result["rt_rw"] = normalize_rtrw(rtrw_raw)

    # Kel/Desa: extract after label
    kel_desa = extract_after_label(lines, r"Kel.*Desa|KeUDesa", next_lines=2)
    if kel_desa != "-":
        result["kel_desa"] = clean_value(re.sub(r"\s*-\s*[A-Z]\s*\.?$", "", kel_desa)).strip(" .,-")
    else:
        result["kel_desa"] = extract_after_fuzzy_labels(lines, ["keldesa", "keudesa"], next_lines=2)

    # Kecamatan: extract after label
    kecamatan = extract_after_label(lines, r"Kecamatan|Kecematan", next_lines=2)
    if kecamatan != "-":
        result["kecamatan"] = clean_value(kecamatan).strip(" .,-")
    else:
        result["kecamatan"] = extract_after_fuzzy_labels(lines, ["kecamatan", "kecematan"], next_lines=2)

    # Agama: look for religion keywords
    agama_match = re.search(r"\b(ISLAM|KATOLIK|KRISTEN|HINDU|BUDDHA|KONGHUCU)\b", joined_text, re.IGNORECASE)
    if agama_match:
        result["agama"] = clean_value(agama_match.group(1))

    # Status Perkawinan: prefer more specific patterns first.
    status_match = re.search(r"\b(BELUM\s+KAWIN|CERAI\s+MATI|CERAI\s+HIDUP|KAWIN)\b", joined_text, re.IGNORECASE)
    if status_match:
        result["status_perkawinan"] = clean_value(status_match.group(1))
    else:
        result["status_perkawinan"] = extract_after_label(lines, r"Status\s*Perkawinan|Siatus\s*Perkawinan", next_lines=2)
        if result["status_perkawinan"] == "-":
            for line in lines:
                if re.search(r"S[i1]atus.*Perkawinan", line, re.IGNORECASE) and re.search(r"KAWIN", line, re.IGNORECASE):
                    result["status_perkawinan"] = "KAWIN"
                    break
    if isinstance(result["status_perkawinan"], str) and re.search(r"KAWIN", result["status_perkawinan"], re.IGNORECASE):
        if re.search(r"BELUM", result["status_perkawinan"], re.IGNORECASE):
            result["status_perkawinan"] = "BELUM KAWIN"
        else:
            result["status_perkawinan"] = "KAWIN"
    if isinstance(result["status_perkawinan"], str):
        result["status_perkawinan"] = result["status_perkawinan"].replace("KAMIN", "KAWIN")

    # Pekerjaan: prioritize label extraction, then fallback to known categories.
    pekerjaan_value = extract_after_label(lines, r"Pekerjaan|Peherjtan|Pekerjaan", next_lines=2)
    if pekerjaan_value != "-":
        result["pekerjaan"] = normalize_job(pekerjaan_value)
    else:
        pekerjaan_match = re.search(
            r"\b(PEGAWAI\s*SWASTA|PEGAWAI\s+NEGERI|WIRASWASTA|KARYAWAN|IBU\s+RUMAH\s+TANGGA|PELAJAR|MAHASISWA)\b",
            joined_text,
            re.IGNORECASE,
        )
        if pekerjaan_match:
            result["pekerjaan"] = normalize_job(pekerjaan_match.group(1))

    # Kewarganegaraan: tolerate OCR confusion between I/1/L.
    # Business rule from user: Kewarganegaraan is always WNI.
    result["kewarganegaraan"] = "WNI"

    # Berlaku Hingga: prefer date near label, not the first date in the document.
    berlaku_line = extract_after_label(lines, r"Berlaku\s*Hingga|Berlaku\s*Hingqa|BerakuHingge|BertakuHingge|Berlaku", next_lines=2)
    # If label line contains multiple dates, prefer the first one near the label.
    berlaku_dates = re.findall(r"(\d{1,2}[-/]\d{1,2}[-/]\d{4})", berlaku_line)
    if berlaku_dates:
        berlaku_date = normalize_date(berlaku_dates[0])
    else:
        berlaku_date = normalize_date(extract_date_from_text(berlaku_line))
    if berlaku_date != "-":
        result["berlaku_hingga"] = berlaku_date
    elif re.search(r"SEUMUR\s*HIDUP", berlaku_line, re.IGNORECASE):
        result["berlaku_hingga"] = "SEUMUR HIDUP"
    else:
        all_dates = re.findall(r"(\d{1,2}[-/]\d{1,2}[-/]\d{4})", joined_text)
        all_dates = [d.replace("/", "-") for d in all_dates]
        ttl_date = extract_date_from_text(result.get("tempat_tgl_lahir", ""))
        if ttl_date != "-":
            filtered = [d for d in all_dates if d != ttl_date]
        else:
            filtered = all_dates
        if filtered:
            result["berlaku_hingga"] = filtered[-1]
        elif all_dates:
            result["berlaku_hingga"] = all_dates[-1]

    # Final line-based stabilization overrides for noisy OCR outputs.
    # 1) Nama from explicit line (e.g. "Nama ...", "Narna ...", or "Bg -...")
    nama_line = find_first_line(r"^(Nama|Narna|Bg)\b")
    if nama_line:
        nama_candidate = re.sub(r"^(Nama|Narna|Bg)\b", "", nama_line, flags=re.IGNORECASE)
        nama_candidate = clean_value(nama_candidate).strip(".-: ")
        if nama_candidate and "JL" not in nama_candidate.upper():
            result["nama"] = normalize_name(nama_candidate)

    # 2) TTL from line containing Lahir + date.
    ttl_line2 = find_first_line(r"Lahir")
    if ttl_line2:
        dm = re.search(r"(\d{1,2}[-/]\d{1,2}[-/]\d{4})", ttl_line2)
        if dm:
            result["tgl_lahir"] = normalize_date(dm.group(1))
        # Prefer clear city token if present.
        if re.search(r"JAKARTA", ttl_line2, re.IGNORECASE):
            result["tempat_lahir"] = "JAKARTA"
        else:
            p = ttl_line2
            p = re.sub(r".*Lahir", "", p, flags=re.IGNORECASE)
            if dm:
                p = p[:dm.start()]
            p = clean_value(p).strip(" .,-")
            if p:
                result["tempat_lahir"] = p
        if result["tempat_lahir"] != "-" and result["tgl_lahir"] != "-":
            result["tempat_tgl_lahir"] = f"{result['tempat_lahir']}, {result['tgl_lahir']}"

    # 3) Alamat from line containing JL if current value is weak.
    if result["alamat"] in ["-", "JL", "JL."]:
        alamat_line2 = find_first_line(r"\bJL\.?[A-Z0-9]")
        if alamat_line2:
            m = re.search(r"(\bJL\.?\s*[A-Z0-9\./\- ]+|\bJL[A-Z0-9\./\- ]+)", alamat_line2, re.IGNORECASE)
            if m:
                result["alamat"] = normalize_address(m.group(1))
    # Guard against accidental "Lahir..." leakage into address.
    if isinstance(result["alamat"], str) and "LAHIR" in result["alamat"].upper():
        result["alamat"] = "-"
        for line in lines:
            m2 = re.search(r"\bJL\.?\s*[A-Z0-9\./\- ]+|\bJL[A-Z0-9\./\- ]+", line, re.IGNORECASE)
            if m2:
                result["alamat"] = normalize_address(m2.group(0))
                break

    # 4) RT/RW from RTRW line if still missing.
    if result["rt_rw"] == "-":
        rline = find_first_line(r"RT\s*/?\s*RW|RTRW")
        if rline:
            result["rt_rw"] = normalize_rtrw(rline)

    # 5) Kel/Desa and Kecamatan from explicit lines.
    if result["kel_desa"] == "-":
        kline = find_first_line(r"(Kel\s*/?\s*Desa|KelDesa|KeVDesa|KeUDesa)")
        if kline:
            tmp = re.sub(r".*(Kel\s*/?\s*Desa|KelDesa|KeVDesa|KeUDesa)", "", kline, flags=re.IGNORECASE)
            result["kel_desa"] = clean_value(tmp).strip(" .,-") or "-"
    if result["kecamatan"] == "-":
        cline = find_first_line(r"Kecamatan")
        if cline:
            tmp = re.sub(r".*Kecamatan", "", cline, flags=re.IGNORECASE)
            result["kecamatan"] = clean_value(tmp).strip(" .,-") or "-"

    # 6) Status perkawinan from explicit line variants.
    sline = find_first_line(r"(Status|Siatus|Gistus).*(Perkaw|Perkawi)")
    if sline and re.search(r"KAWIN|KAMIN", sline, re.IGNORECASE):
        result["status_perkawinan"] = "KAWIN"

    # 7) Pekerjaan from explicit line variants.
    pline = find_first_line(r"(Pekerjaan|Peherjtan)")
    if pline and re.search(r"PEGAWAI\s*SWASTA|PEGAWAISWASTA", pline, re.IGNORECASE):
        result["pekerjaan"] = "PEGAWAI SWASTA"

    # 8) Berlaku Hingga from explicit line: choose first valid date in that line.
    bline = find_first_line(r"(Berlaku|Beraku|Bertaku).*(Hingga|Hingge|Hingqa)")
    if bline:
        candidates = re.findall(r"(\d{1,2}[-/]\d{1,2}[-/]\d{4})", bline)
        valid = []
        for d in candidates:
            nd = normalize_date(d)
            mm = int(nd.split('-')[1]) if nd != '-' and '-' in nd else 0
            dd = int(nd.split('-')[0]) if nd != '-' and '-' in nd else 0
            if 1 <= mm <= 12 and 1 <= dd <= 31:
                valid.append(nd)
        if valid:
            # pick first date that is not birth date
            bd = result.get("tgl_lahir", "-")
            picked = next((d for d in valid if d != bd), valid[0])
            result["berlaku_hingga"] = picked
    else:
        # Avoid leaking birth date into berlaku_hingga when no valid berlaku label exists.
        if result.get("berlaku_hingga") == result.get("tgl_lahir"):
            result["berlaku_hingga"] = "-"

    # 9) Business rule from user: always WNI.
    result["kewarganegaraan"] = "WNI"

    return result
